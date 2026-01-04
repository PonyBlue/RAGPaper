from typing import List, Optional, Tuple, Dict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    CHROMA_DB_DIR, EMBEDDING_MODEL, COLLECTION_NAME, TOP_K,
    USE_RERANK, RERANK_MODEL, FETCH_K,
    USE_HYBRID, HYBRID_ALPHA, BM25_K1, BM25_B
)


class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        persist_directory: str = str(CHROMA_DB_DIR),
        collection_name: str = COLLECTION_NAME,
        use_rerank: bool = USE_RERANK,
        rerank_model: str = RERANK_MODEL,
        use_hybrid: bool = USE_HYBRID,
        hybrid_alpha: float = HYBRID_ALPHA
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_rerank = use_rerank
        self.rerank_model = rerank_model
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha

        # 初始化Embedding模型
        print(f"正在加载Embedding模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding模型加载完成")

        self.vectorstore: Optional[Chroma] = None
        self.reranker = None
        self.hybrid_retriever = None

        # 初始化Reranker（如果启用）
        if self.use_rerank:
            self._init_reranker()

        # 注意：混合检索需要在加载vectorstore后初始化
        # 因为需要获取所有文档

    def _init_reranker(self):
        """初始化Reranker"""
        try:
            from src.reranker import create_reranker
            print(f"[Rerank] 启用Rerank功能，模型: {self.rerank_model}")
            self.reranker = create_reranker(use_model=True, model_name=self.rerank_model)
        except Exception as e:
            print(f"[警告] Reranker初始化失败，将不使用Rerank: {e}")
            self.reranker = None
            self.use_rerank = False

    def _clean_text(self, text: str) -> str:
        """清理文本中的问题字符"""
        import unicodedata

        # 1. 去除首尾空白
        text = text.strip()

        # 2. 规范化 Unicode 字符 (NFC normalization)
        text = unicodedata.normalize('NFC', text)

        # 3. 确保文本可以被正确编码
        # 这一步将移除或替换无法正确处理的字符
        try:
            text.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # 如果编码失败，使用errors='ignore'跳过问题字符
            text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

        return text

    def _validate_documents(self, documents: List[Document]) -> List[Document]:
        """验证并清理文档列表"""
        valid_docs = []
        skipped = 0

        for i, doc in enumerate(documents):
            try:
                # 检查是否是Document对象
                if not isinstance(doc, Document):
                    print(f"⚠️ 第{i}个不是Document对象，类型: {type(doc)}")
                    skipped += 1
                    continue

                # 检查page_content是否存在
                if not hasattr(doc, 'page_content'):
                    print(f"⚠️ 第{i}个Document没有page_content属性")
                    skipped += 1
                    continue

                # 检查page_content是否是字符串
                if not isinstance(doc.page_content, str):
                    print(f"⚠️ 第{i}个page_content不是字符串，类型: {type(doc.page_content)}")
                    skipped += 1
                    continue

                # 检查是否为空
                if not doc.page_content or not doc.page_content.strip():
                    skipped += 1
                    continue

                # 检查长度是否合理（至少10个字符）
                if len(doc.page_content.strip()) < 10:
                    skipped += 1
                    continue

                # 深度清理文本
                doc.page_content = self._clean_text(doc.page_content)

                # 再次检查清理后的文本
                if not doc.page_content or len(doc.page_content) < 10:
                    skipped += 1
                    continue

                # 确保metadata存在且是字典
                if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                    doc.metadata = {}

                # 清理metadata中的值，确保都是基本类型
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        cleaned_metadata[key] = value
                    else:
                        # 将非基本类型转换为字符串
                        cleaned_metadata[key] = str(value)

                doc.metadata = cleaned_metadata

                valid_docs.append(doc)

            except Exception as e:
                print(f"⚠️ 验证第{i}个文档时出错: {e}")
                skipped += 1
                continue

        if skipped > 0:
            print(f"⚠️ 跳过了 {skipped} 个无效文档块 (总共{len(documents)}个)")

        return valid_docs

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """创建新的向量数据库"""
        # 验证文档
        documents = self._validate_documents(documents)

        if not documents:
            raise ValueError("没有有效的文档可以创建向量数据库")

        print(f"正在创建向量数据库，文档块数量: {len(documents)}")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        print(f"向量数据库创建完成，已持久化至: {self.persist_directory}")
        return self.vectorstore

    def load_vectorstore(self) -> Optional[Chroma]:
        """加载已存在的向量数据库"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print("向量数据库加载成功")

            # 初始化混合检索（如果启用）
            if self.use_hybrid:
                self._init_hybrid_retriever()

            return self.vectorstore
        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}")
            return None

    def add_documents(self, documents: List[Document]):
        """向现有向量数据库添加文档"""
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化，请先创建或加载")

        # 验证文档
        documents = self._validate_documents(documents)

        if not documents:
            print("⚠️ 没有有效的文档可以添加")
            return

        self.vectorstore.add_documents(documents)
        print(f"已添加 {len(documents)} 个文档块到向量数据库")

    def similarity_search(self, query: str, k: int = TOP_K) -> List[Document]:
        """相似度搜索"""
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def similarity_search_with_score(self, query: str, k: int = TOP_K) -> List[tuple]:
        """带分数的相似度搜索"""
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def get_retriever(self, k: int = TOP_K):
        """获取检索器对象"""
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def similarity_search_with_rerank(
        self,
        query: str,
        k: int = TOP_K,
        fetch_k: int = FETCH_K
    ) -> List[Document]:
        """
        带Rerank的相似度搜索

        Args:
            query: 查询文本
            k: 最终返回的文档数
            fetch_k: 初步检索的文档数（用于rerank前的候选集）

        Returns:
            重排序后的前k个文档
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        # 如果未启用Rerank，直接返回普通检索结果
        if not self.use_rerank or self.reranker is None:
            return self.similarity_search(query, k=k)

        # 1. 初步检索：获取更多候选文档
        candidates = self.vectorstore.similarity_search(query, k=fetch_k)

        if not candidates:
            return []

        # 2. Rerank：重新排序
        reranked_results = self.reranker.rerank(query, candidates, top_k=k)

        # 3. 返回重排序后的文档（不包含分数）
        return [doc for doc, score in reranked_results]

    def similarity_search_with_rerank_and_scores(
        self,
        query: str,
        k: int = TOP_K,
        fetch_k: int = FETCH_K
    ) -> List[tuple]:
        """
        带Rerank和分数的相似度搜索

        Returns:
            [(Document, score), ...] 重排序后的文档和分数对
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        # 如果未启用Rerank，返回普通检索结果和分数
        if not self.use_rerank or self.reranker is None:
            return self.similarity_search_with_score(query, k=k)

        # 1. 初步检索
        candidates = self.vectorstore.similarity_search(query, k=fetch_k)

        if not candidates:
            return []

        # 2. Rerank并返回分数
        return self.reranker.rerank(query, candidates, top_k=k)

    def get_retriever_with_rerank(self, k: int = TOP_K, fetch_k: int = FETCH_K):
        """
        获取带Rerank的检索器

        这个检索器会自动使用Rerank进行重排序
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        # 如果未启用Rerank，返回普通检索器
        if not self.use_rerank:
            return self.get_retriever(k=k)

        # 创建自定义检索器包装类
        class RerankRetriever:
            def __init__(self, vs_manager, k, fetch_k):
                self.vs_manager = vs_manager
                self.k = k
                self.fetch_k = fetch_k

            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.vs_manager.similarity_search_with_rerank(
                    query, k=self.k, fetch_k=self.fetch_k
                )

            def invoke(self, query: str) -> List[Document]:
                return self.get_relevant_documents(query)

        return RerankRetriever(self, k, fetch_k)

    def delete_collection(self):
        """删除集合"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            print("向量数据库集合已删除")

    # ==================== 混合检索相关方法 ====================

    def get_all_documents(self) -> List[Document]:
        """
        获取向量数据库中的所有文档
        用于初始化BM25索引

        Returns:
            文档列表
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化")

        try:
            # 获取集合中的所有文档
            # 使用一个足够大的k值来获取所有文档
            all_docs = self.vectorstore.similarity_search("", k=10000)
            print(f"[混合检索] 获取到 {len(all_docs)} 个文档用于BM25索引")
            return all_docs
        except Exception as e:
            print(f"[错误] 获取所有文档失败: {e}")
            # Fallback: 使用_collection直接访问
            try:
                collection = self.vectorstore._collection
                results = collection.get()
                docs = []
                for i in range(len(results['ids'])):
                    doc = Document(
                        page_content=results['documents'][i],
                        metadata=results['metadatas'][i] if results['metadatas'] else {}
                    )
                    docs.append(doc)
                print(f"[混合检索] 通过collection获取到 {len(docs)} 个文档")
                return docs
            except Exception as e2:
                print(f"[错误] Fallback方法也失败: {e2}")
                return []

    def _init_hybrid_retriever(self):
        """初始化混合检索器"""
        try:
            from src.hybrid_retriever import HybridRetriever

            print(f"[混合检索] 启用混合检索功能")
            print(f"  语义检索权重(alpha): {self.hybrid_alpha:.2f}")
            print(f"  BM25检索权重: {1-self.hybrid_alpha:.2f}")

            # 获取所有文档
            all_documents = self.get_all_documents()

            if not all_documents:
                print("[警告] 无法获取文档，混合检索初始化失败")
                self.use_hybrid = False
                return

            # 创建混合检索器
            self.hybrid_retriever = HybridRetriever(
                vectorstore=self.vectorstore,
                documents=all_documents,
                alpha=self.hybrid_alpha,
                k1=BM25_K1,
                b=BM25_B
            )

            print("[OK] 混合检索器初始化完成")

        except ImportError as e:
            print(f"[警告] 混合检索模块导入失败: {e}")
            print("      请确保已安装rank-bm25: pip install rank-bm25")
            self.use_hybrid = False
            self.hybrid_retriever = None
        except Exception as e:
            print(f"[警告] 混合检索器初始化失败: {e}")
            self.use_hybrid = False
            self.hybrid_retriever = None

    def hybrid_search(
        self,
        query: str,
        k: int = TOP_K,
        fetch_k: int = FETCH_K
    ) -> List[Document]:
        """
        混合检索

        Args:
            query: 查询文本
            k: 最终返回的文档数
            fetch_k: 从每个检索器获取的候选数

        Returns:
            混合检索结果（文档列表）
        """
        if not self.use_hybrid or self.hybrid_retriever is None:
            # 如果未启用混合检索，使用普通检索
            return self.similarity_search(query, k=k)

        # 执行混合检索
        results = self.hybrid_retriever.hybrid_search(
            query=query,
            k=k,
            fetch_k=fetch_k
        )

        # 返回文档（不包含分数）
        return [doc for doc, score in results]

    def hybrid_search_with_scores(
        self,
        query: str,
        k: int = TOP_K,
        fetch_k: int = FETCH_K
    ) -> List[Tuple[Document, float]]:
        """
        带分数的混合检索

        Returns:
            [(Document, score), ...] 文档和融合分数对
        """
        if not self.use_hybrid or self.hybrid_retriever is None:
            # 如果未启用混合检索，使用普通检索
            return self.similarity_search_with_score(query, k=k)

        # 执行混合检索
        return self.hybrid_retriever.hybrid_search(
            query=query,
            k=k,
            fetch_k=fetch_k
        )

    def hybrid_search_with_details(
        self,
        query: str,
        k: int = TOP_K,
        fetch_k: int = FETCH_K
    ) -> Dict:
        """
        带详细信息的混合检索

        Returns:
            包含混合结果、语义结果、BM25结果的字典
        """
        if not self.use_hybrid or self.hybrid_retriever is None:
            raise ValueError("混合检索未启用")

        return self.hybrid_retriever.search_with_details(
            query=query,
            k=k,
            fetch_k=fetch_k
        )

    def get_retriever_with_hybrid(self, k: int = TOP_K, fetch_k: int = FETCH_K):
        """
        获取混合检索器

        Returns:
            混合检索器实例
        """
        if not self.use_hybrid:
            # 如果未启用，返回普通检索器
            return self.get_retriever(k=k)

        # 创建混合检索器包装类
        class HybridRetrieverWrapper:
            def __init__(self, vs_manager, k, fetch_k):
                self.vs_manager = vs_manager
                self.k = k
                self.fetch_k = fetch_k

            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.vs_manager.hybrid_search(
                    query, k=self.k, fetch_k=self.fetch_k
                )

            def invoke(self, query: str) -> List[Document]:
                return self.get_relevant_documents(query)

        return HybridRetrieverWrapper(self, k, fetch_k)

