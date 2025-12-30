from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL, COLLECTION_NAME, TOP_K


class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        persist_directory: str = str(CHROMA_DB_DIR),
        collection_name: str = COLLECTION_NAME
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 初始化Embedding模型
        print(f"正在加载Embedding模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding模型加载完成")

        self.vectorstore: Optional[Chroma] = None

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

    def delete_collection(self):
        """删除集合"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            print("向量数据库集合已删除")
