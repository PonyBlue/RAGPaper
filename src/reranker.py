"""
Rerank重排序模块
使用Cross-Encoder对初步检索结果进行精排，提升检索准确率
"""

from typing import List, Tuple
from langchain_core.documents import Document


class Reranker:
    """
    重排序器
    使用Cross-Encoder模型对检索结果进行重新排序
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        """
        初始化Reranker

        Args:
            model_name: Rerank模型名称
                - "BAAI/bge-reranker-base": 中等大小，平衡性能和速度
                - "BAAI/bge-reranker-large": 大模型，效果最好但速度慢
                - "BAAI/bge-reranker-v2-m3": 多语言支持
            device: 运行设备 ("cpu" 或 "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model = None

        print(f"[初始化] 准备加载Rerank模型: {model_name}")
        self._load_model()

    def _load_model(self):
        """延迟加载模型（按需加载）"""
        try:
            from sentence_transformers import CrossEncoder
            print(f"[加载中] 正在加载Rerank模型，这可能需要一些时间...")
            self.model = CrossEncoder(self.model_name, device=self.device)
            print(f"[OK] Rerank模型加载完成: {self.model_name}")
        except ImportError:
            print("[错误] 未安装sentence-transformers库")
            print("      请运行: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            print(f"      模型名称: {self.model_name}")
            print("      请检查模型名称是否正确，或网络连接是否正常")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        重排序文档

        Args:
            query: 查询问题
            documents: 候选文档列表
            top_k: 返回前K个结果

        Returns:
            [(document, score), ...] 按分数降序排列的文档和分数对
        """
        if not documents:
            return []

        if self.model is None:
            print("[警告] Rerank模型未加载，返回原始顺序")
            return [(doc, 0.0) for doc in documents[:top_k]]

        # 准备输入对：(query, document_text)
        pairs = [[query, doc.page_content] for doc in documents]

        # 计算相关性分数
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"[错误] Rerank评分失败: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]

        # 组合文档和分数
        doc_scores = list(zip(documents, scores))

        # 按分数降序排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前K个
        return doc_scores[:top_k]

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        重排序文档（带分数阈值过滤）

        Args:
            query: 查询问题
            documents: 候选文档列表
            top_k: 返回前K个结果
            score_threshold: 分数阈值，低于此分数的文档将被过滤

        Returns:
            [(document, score), ...] 过滤后按分数降序排列的文档
        """
        # 先进行重排序
        reranked = self.rerank(query, documents, top_k=len(documents))

        # 过滤低分文档
        filtered = [(doc, score) for doc, score in reranked if score >= score_threshold]

        # 返回前K个
        return filtered[:top_k]


class SimpleReranker:
    """
    简单的重排序器（不依赖外部模型）
    基于关键词匹配进行重排序，可作为Fallback方案
    """

    def __init__(self):
        print("[初始化] 使用简单重排序器（基于关键词匹配）")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        基于关键词匹配的简单重排序

        Args:
            query: 查询问题
            documents: 候选文档列表
            top_k: 返回前K个结果

        Returns:
            [(document, score), ...] 按分数降序排列
        """
        if not documents:
            return []

        # 提取查询中的关键词
        query_words = set(query.lower().split())

        # 计算每个文档的匹配分数
        doc_scores = []
        for doc in documents:
            content_words = set(doc.page_content.lower().split())

            # 计算交集
            overlap = len(query_words & content_words)

            # 归一化分数
            score = overlap / len(query_words) if query_words else 0.0

            doc_scores.append((doc, score))

        # 按分数降序排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回前K个
        return doc_scores[:top_k]


def create_reranker(use_model: bool = True, model_name: str = "BAAI/bge-reranker-base"):
    """
    创建Reranker实例的工厂函数

    Args:
        use_model: 是否使用模型，False则使用简单匹配
        model_name: 模型名称

    Returns:
        Reranker或SimpleReranker实例
    """
    if use_model:
        try:
            return Reranker(model_name=model_name)
        except Exception as e:
            print(f"[警告] 无法加载Rerank模型，降级使用简单重排序: {e}")
            return SimpleReranker()
    else:
        return SimpleReranker()


# 测试代码
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Reranker模块测试")
    print("=" * 70)

    # 创建测试文档
    test_docs = [
        Document(
            page_content="QMDF是一个分子表示学习方法，使用图神经网络",
            metadata={"source": "QMDF.pdf", "page": 1}
        ),
        Document(
            page_content="point2skeleton提出了点云骨架提取算法",
            metadata={"source": "point2skeleton.pdf", "page": 2}
        ),
        Document(
            page_content="实验在ModelNet40数据集上进行，准确率达到92%",
            metadata={"source": "QMDF.pdf", "page": 5}
        ),
        Document(
            page_content="图神经网络在分子性质预测中表现优异",
            metadata={"source": "QMDF.pdf", "page": 3}
        ),
    ]

    query = "QMDF使用了什么方法?"

    # 测试简单重排序器
    print("\n[测试] 简单重排序器:")
    simple_reranker = SimpleReranker()
    simple_results = simple_reranker.rerank(query, test_docs, top_k=3)

    for i, (doc, score) in enumerate(simple_results, 1):
        print(f"  {i}. [分数: {score:.3f}] {doc.page_content[:50]}...")

    # 测试模型重排序器（如果可用）
    print("\n[测试] 模型重排序器:")
    print("  (如果未安装sentence-transformers，将跳过)")

    try:
        model_reranker = Reranker(model_name="BAAI/bge-reranker-base")
        model_results = model_reranker.rerank(query, test_docs, top_k=3)

        for i, (doc, score) in enumerate(model_results, 1):
            print(f"  {i}. [分数: {score:.3f}] {doc.page_content[:50]}...")

        print("\n[OK] 模型重排序测试成功!")

    except Exception as e:
        print(f"  [跳过] 无法测试模型重排序: {e}")

    print("\n" + "=" * 70)
