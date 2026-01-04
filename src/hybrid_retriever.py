"""
混合检索模块（Hybrid Retrieval）
结合语义检索和BM25关键词检索，提升检索准确率

技术方案:
- 语义检索: 基于向量相似度（SBERT Embeddings）
- BM25检索: 基于关键词匹配（统计方法）
- 融合策略: 加权组合 (α × semantic + (1-α) × BM25)
"""

from typing import List, Tuple, Dict, Optional
from langchain_core.documents import Document
import numpy as np


class BM25Retriever:
    """
    BM25检索器
    基于Okapi BM25算法的关键词检索
    """

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器

        Args:
            documents: 文档列表
            k1: BM25参数，控制词频饱和度（默认1.5）
            b: BM25参数，控制文档长度归一化（默认0.75）
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_corpus = []

        print(f"[BM25] 初始化BM25检索器，文档数: {len(documents)}")
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """简单的分词"""
        # 转小写并按空格分词
        tokens = text.lower().split()
        # 过滤掉太短的词
        tokens = [t for t in tokens if len(t) > 2]
        return tokens

    def _build_index(self):
        """构建BM25索引"""
        try:
            from rank_bm25 import BM25Okapi

            print("[BM25] 构建BM25索引...")

            # 对所有文档进行分词
            self.tokenized_corpus = [
                self._tokenize(doc.page_content)
                for doc in self.documents
            ]

            # 构建BM25索引
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b
            )

            print(f"[OK] BM25索引构建完成")

        except ImportError:
            print("[错误] 未安装rank-bm25库")
            print("      请运行: pip install rank-bm25")
            raise

    def search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        BM25检索

        Args:
            query: 查询文本
            k: 返回Top-K结果

        Returns:
            [(Document, score), ...] 按BM25分数降序排列
        """
        if self.bm25 is None:
            print("[警告] BM25索引未构建")
            return []

        # 分词查询
        tokenized_query = self._tokenize(query)

        # 计算BM25分数
        scores = self.bm25.get_scores(tokenized_query)

        # 组合文档和分数
        doc_scores = list(zip(self.documents, scores))

        # 按分数降序排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回Top-K
        return doc_scores[:k]

    def get_top_k_indices(self, query: str, k: int = 4) -> List[int]:
        """
        获取Top-K文档的索引

        Returns:
            文档索引列表
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取Top-K索引
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices.tolist()


class HybridRetriever:
    """
    混合检索器
    结合语义检索和BM25检索，通过加权融合提升准确率
    """

    def __init__(
        self,
        vectorstore,
        documents: List[Document],
        alpha: float = 0.7,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        初始化混合检索器

        Args:
            vectorstore: ChromaDB向量数据库实例
            documents: 所有文档列表（用于BM25）
            alpha: 语义检索权重（0-1），BM25权重为1-alpha
            k1, b: BM25参数
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.alpha = alpha

        print(f"[混合检索] 初始化混合检索器")
        print(f"  语义检索权重: {alpha:.2f}")
        print(f"  BM25检索权重: {1-alpha:.2f}")

        # 初始化BM25检索器
        self.bm25_retriever = BM25Retriever(documents, k1=k1, b=b)

        # 创建文档ID到文档的映射
        self._build_doc_mapping()

    def _build_doc_mapping(self):
        """构建文档映射（用于去重和查找）"""
        self.doc_map = {}
        for i, doc in enumerate(self.documents):
            # 使用内容的前100个字符作为唯一标识
            doc_key = doc.page_content[:100]
            self.doc_map[doc_key] = (i, doc)

    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        """
        归一化分数到[0, 1]区间

        Args:
            scores: 原始分数列表

        Returns:
            归一化后的分数数组
        """
        scores = np.array(scores)

        # 避免除零错误
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores)

        # Min-Max归一化
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20
    ) -> List[Tuple[Document, float]]:
        """
        混合检索

        Args:
            query: 查询文本
            k: 最终返回的文档数
            fetch_k: 从每个检索器获取的候选文档数

        Returns:
            [(Document, score), ...] 按融合分数降序排列
        """
        print(f"\n[混合检索] 执行混合检索: {query[:50]}...")

        # 1. 语义检索
        print(f"  [1/3] 语义检索 (Top-{fetch_k})...")
        semantic_results = self.vectorstore.similarity_search_with_score(
            query, k=fetch_k
        )

        # 提取文档和分数
        semantic_docs = [doc for doc, score in semantic_results]
        # ChromaDB返回的是距离，需要转换为相似度（距离越小越相似）
        semantic_scores = [1.0 / (1.0 + score) for doc, score in semantic_results]

        # 归一化语义分数
        semantic_scores_norm = self._normalize_scores(semantic_scores)

        # 2. BM25检索
        print(f"  [2/3] BM25检索 (Top-{fetch_k})...")
        bm25_results = self.bm25_retriever.search(query, k=fetch_k)
        bm25_docs = [doc for doc, score in bm25_results]
        bm25_scores = [score for doc, score in bm25_results]

        # 归一化BM25分数
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # 3. 融合分数
        print(f"  [3/3] 融合分数 (α={self.alpha:.2f})...")

        # 创建所有候选文档的分数字典
        doc_scores = {}

        # 添加语义检索分数
        for doc, score in zip(semantic_docs, semantic_scores_norm):
            doc_key = doc.page_content[:100]
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {"doc": doc, "semantic": 0.0, "bm25": 0.0}
            doc_scores[doc_key]["semantic"] = score

        # 添加BM25分数
        for doc, score in zip(bm25_docs, bm25_scores_norm):
            doc_key = doc.page_content[:100]
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {"doc": doc, "semantic": 0.0, "bm25": 0.0}
            doc_scores[doc_key]["bm25"] = score

        # 计算融合分数
        final_results = []
        for doc_key, info in doc_scores.items():
            # 融合分数 = α × semantic + (1-α) × BM25
            hybrid_score = (
                self.alpha * info["semantic"] +
                (1 - self.alpha) * info["bm25"]
            )
            final_results.append((info["doc"], hybrid_score))

        # 按融合分数降序排序
        final_results.sort(key=lambda x: x[1], reverse=True)

        # 返回Top-K
        top_k = final_results[:k]

        print(f"[OK] 混合检索完成，返回 {len(top_k)} 个结果")
        return top_k

    def search_with_details(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20
    ) -> Dict:
        """
        混合检索（带详细信息）

        Returns:
            {
                "results": [(Document, hybrid_score), ...],
                "semantic_results": [...],
                "bm25_results": [...],
                "details": [...]  # 每个文档的详细分数
            }
        """
        # 执行混合检索
        results = self.hybrid_search(query, k=k, fetch_k=fetch_k)

        # 同时返回单独的检索结果（用于分析对比）
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k)
        bm25_results = self.bm25_retriever.search(query, k=k)

        return {
            "results": results,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "alpha": self.alpha
        }


class SimpleBM25Retriever:
    """
    简单的BM25检索器（不依赖rank-bm25库）
    基于TF-IDF的简化实现
    """

    def __init__(self, documents: List[Document]):
        self.documents = documents
        print(f"[SimpleBM25] 使用简化BM25实现，文档数: {len(documents)}")

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """简单的关键词匹配"""
        query_words = set(self._tokenize(query))

        doc_scores = []
        for doc in self.documents:
            doc_words = set(self._tokenize(doc.page_content))
            # 计算交集大小作为分数
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words) if query_words else 0.0
            doc_scores.append((doc, score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:k]


def create_hybrid_retriever(
    vectorstore,
    documents: List[Document],
    alpha: float = 0.7,
    use_advanced_bm25: bool = True
) -> HybridRetriever:
    """
    创建混合检索器的工厂函数

    Args:
        vectorstore: 向量数据库
        documents: 文档列表
        alpha: 语义检索权重
        use_advanced_bm25: 是否使用高级BM25（需要rank-bm25库）

    Returns:
        HybridRetriever实例
    """
    try:
        if use_advanced_bm25:
            return HybridRetriever(vectorstore, documents, alpha=alpha)
    except ImportError:
        print("[警告] rank-bm25库未安装，使用简化版BM25")

    # Fallback：使用简化版
    # （这里需要修改HybridRetriever以支持简化BM25）
    return HybridRetriever(vectorstore, documents, alpha=alpha)
