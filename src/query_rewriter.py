"""
查询改写模块（Query Rewriting）
使用LLM将复杂查询改写为更优的形式，提升检索效果

策略:
1. 查询拆解: 将复杂问题拆解为多个子问题
2. 查询扩展: 添加同义词和相关术语
3. 查询简化: 提取核心查询意图
"""

from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from src.config import OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME


class QueryRewriter:
    """
    查询改写器
    使用LLM对查询进行改写和优化
    """

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        api_base: str = OPENAI_API_BASE,
        model_name: str = MODEL_NAME,
        temperature: float = 0.3
    ):
        """
        初始化查询改写器

        Args:
            api_key: OpenAI API密钥
            api_base: API基础URL
            model_name: 模型名称
            temperature: 温度参数（低温度=更确定性）
        """
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model=model_name,
            temperature=temperature
        )
        print("[查询改写] 初始化完成")

    def decompose_query(self, query: str, max_sub_queries: int = 3) -> List[str]:
        """
        查询拆解: 将复杂问题拆解为多个子问题

        Args:
            query: 原始查询
            max_sub_queries: 最大子查询数量

        Returns:
            子查询列表
        """
        prompt = f"""你是一个查询分析专家。请将以下复杂问题拆解为{max_sub_queries}个更简单、更具体的子问题。

原问题: {query}

要求:
1. 每个子问题应该聚焦一个具体方面
2. 子问题之间应该相互独立
3. 子问题的答案组合起来能回答原问题
4. 每行一个子问题，不要编号

子问题:"""

        try:
            response = self.llm.invoke(prompt)
            sub_queries = response.content.strip().split('\n')

            # 清理子查询
            sub_queries = [
                q.strip().lstrip('-').lstrip('•').lstrip('*').strip()
                for q in sub_queries
                if q.strip()
            ]

            # 限制数量
            sub_queries = sub_queries[:max_sub_queries]

            print(f"[查询拆解] {query[:50]}... → {len(sub_queries)}个子查询")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")

            return sub_queries

        except Exception as e:
            print(f"[警告] 查询拆解失败: {e}")
            return [query]  # Fallback: 返回原查询

    def expand_query(self, query: str) -> List[str]:
        """
        查询扩展: 添加同义词和相关术语

        Args:
            query: 原始查询

        Returns:
            扩展后的查询列表（包括原查询）
        """
        prompt = f"""你是一个查询扩展专家。请为以下查询生成2-3个变体，使用不同的表述方式或同义词。

原查询: {query}

要求:
1. 保持原意不变
2. 使用不同的词汇表达相同的意思
3. 可以包含专业术语的通俗说法
4. 每行一个变体查询

变体查询:"""

        try:
            response = self.llm.invoke(prompt)
            expanded_queries = response.content.strip().split('\n')

            # 清理并添加原查询
            expanded_queries = [
                q.strip().lstrip('-').lstrip('•').lstrip('*').strip()
                for q in expanded_queries
                if q.strip()
            ]

            # 始终包含原查询
            if query not in expanded_queries:
                expanded_queries.insert(0, query)

            print(f"[查询扩展] {query[:50]}... → {len(expanded_queries)}个变体")

            return expanded_queries[:4]  # 最多4个变体

        except Exception as e:
            print(f"[警告] 查询扩展失败: {e}")
            return [query]

    def simplify_query(self, query: str) -> str:
        """
        查询简化: 提取核心查询意图

        Args:
            query: 原始查询（可能冗长或模糊）

        Returns:
            简化后的查询
        """
        prompt = f"""你是一个查询优化专家。请将以下查询简化为一个简洁、明确的问题。

原查询: {query}

要求:
1. 保留核心意图
2. 去除冗余词汇
3. 使用简洁的表述
4. 只返回简化后的查询，不要解释

简化查询:"""

        try:
            response = self.llm.invoke(prompt)
            simplified = response.content.strip()

            print(f"[查询简化] {query[:50]}... → {simplified[:50]}...")

            return simplified

        except Exception as e:
            print(f"[警告] 查询简化失败: {e}")
            return query


class MultiQueryRetriever:
    """
    多查询检索器
    对多个查询并行检索并合并结果
    """

    def __init__(self, vectorstore_manager):
        """
        初始化多查询检索器

        Args:
            vectorstore_manager: VectorStoreManager实例
        """
        self.vs_manager = vectorstore_manager

    def retrieve_multi_queries(
        self,
        queries: List[str],
        k: int = 4,
        strategy: str = "union"
    ) -> List[Document]:
        """
        多查询检索

        Args:
            queries: 查询列表
            k: 每个查询返回的文档数
            strategy: 合并策略 ("union": 并集, "intersection": 交集)

        Returns:
            合并后的文档列表
        """
        print(f"[多查询检索] {len(queries)}个查询，策略: {strategy}")

        all_docs = []
        doc_scores = {}  # 记录文档和分数

        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query[:60]}...")

            try:
                # 检索（使用混合检索或普通检索）
                if self.vs_manager.use_hybrid:
                    results = self.vs_manager.hybrid_search_with_scores(query, k=k)
                else:
                    results = self.vs_manager.similarity_search_with_score(query, k=k)

                # 收集文档和分数
                for doc, score in results:
                    doc_id = self._get_doc_id(doc)

                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {
                            "doc": doc,
                            "scores": [],
                            "queries": []
                        }

                    doc_scores[doc_id]["scores"].append(score)
                    doc_scores[doc_id]["queries"].append(query)

                    all_docs.append(doc)

            except Exception as e:
                print(f"    [错误] 查询失败: {e}")

        # 根据策略合并结果
        if strategy == "union":
            # 并集: 所有出现过的文档
            merged_docs = self._merge_by_union(doc_scores, k=k)
        elif strategy == "intersection":
            # 交集: 只保留在所有查询中都出现的文档
            merged_docs = self._merge_by_intersection(doc_scores, len(queries), k=k)
        else:
            # 默认: 按平均分数排序
            merged_docs = self._merge_by_avg_score(doc_scores, k=k)

        print(f"[OK] 合并后返回 {len(merged_docs)} 个文档")
        return merged_docs

    def _merge_by_union(self, doc_scores: Dict, k: int) -> List[Document]:
        """并集合并: 按平均分数排序"""
        scored_docs = []

        for doc_id, info in doc_scores.items():
            avg_score = sum(info["scores"]) / len(info["scores"])
            scored_docs.append((info["doc"], avg_score))

        # 按平均分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:k]]

    def _merge_by_intersection(
        self,
        doc_scores: Dict,
        num_queries: int,
        k: int
    ) -> List[Document]:
        """交集合并: 只保留在所有查询中都出现的文档"""
        scored_docs = []

        for doc_id, info in doc_scores.items():
            # 只保留在所有查询中都出现的文档
            if len(info["scores"]) >= num_queries:
                avg_score = sum(info["scores"]) / len(info["scores"])
                scored_docs.append((info["doc"], avg_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:k]]

    def _merge_by_avg_score(self, doc_scores: Dict, k: int) -> List[Document]:
        """按平均分数合并（与union相同）"""
        return self._merge_by_union(doc_scores, k)

    def _get_doc_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        # 使用内容的前50个字符作为额外标识
        content_hash = doc.page_content[:50]
        return f"{source}_p{page}_{hash(content_hash)}"


class QueryRewritingRetriever:
    """
    查询改写检索器
    集成查询改写和多查询检索
    """

    def __init__(
        self,
        vectorstore_manager,
        query_rewriter: Optional[QueryRewriter] = None,
        rewrite_strategy: str = "decompose"
    ):
        """
        初始化查询改写检索器

        Args:
            vectorstore_manager: VectorStoreManager实例
            query_rewriter: QueryRewriter实例（可选）
            rewrite_strategy: 改写策略 ("decompose", "expand", "both")
        """
        self.vs_manager = vectorstore_manager
        self.query_rewriter = query_rewriter
        self.rewrite_strategy = rewrite_strategy
        self.multi_retriever = MultiQueryRetriever(vectorstore_manager)

        if query_rewriter:
            print(f"[查询改写检索] 启用查询改写，策略: {rewrite_strategy}")
        else:
            print("[查询改写检索] 未启用查询改写（需要LLM）")

    def retrieve(
        self,
        query: str,
        k: int = 4,
        merge_strategy: str = "union"
    ) -> List[Document]:
        """
        带查询改写的检索

        Args:
            query: 原始查询
            k: 返回的文档数
            merge_strategy: 合并策略

        Returns:
            检索结果
        """
        # 如果没有query_rewriter，使用普通检索
        if not self.query_rewriter:
            return self.vs_manager.similarity_search(query, k=k)

        print(f"\n[查询改写检索] 原始查询: {query}")

        # 根据策略改写查询
        if self.rewrite_strategy == "decompose":
            queries = self.query_rewriter.decompose_query(query)
        elif self.rewrite_strategy == "expand":
            queries = self.query_rewriter.expand_query(query)
        elif self.rewrite_strategy == "both":
            # 先拆解，再扩展每个子查询
            sub_queries = self.query_rewriter.decompose_query(query, max_sub_queries=2)
            queries = []
            for sq in sub_queries:
                expanded = self.query_rewriter.expand_query(sq)
                queries.extend(expanded[:2])  # 每个子查询扩展2个
        else:
            queries = [query]

        # 多查询检索
        results = self.multi_retriever.retrieve_multi_queries(
            queries=queries,
            k=k,
            strategy=merge_strategy
        )

        return results


# ==================== 简化版（不依赖LLM）====================

class SimpleQueryRewriter:
    """
    简单查询改写器
    不依赖LLM，使用规则进行简单改写
    """

    def __init__(self):
        print("[简单查询改写] 使用规则改写（不需要LLM）")

    def decompose_query(self, query: str, max_sub_queries: int = 3) -> List[str]:
        """
        简单拆解: 按标点符号拆分
        """
        # 按句号、问号、分号拆分
        import re
        sub_queries = re.split(r'[。？；]', query)
        sub_queries = [q.strip() for q in sub_queries if q.strip()]

        if len(sub_queries) <= 1:
            # 如果没有拆分成功，返回原查询
            return [query]

        return sub_queries[:max_sub_queries]

    def expand_query(self, query: str) -> List[str]:
        """
        简单扩展: 添加一些变体（有限）
        """
        # 简单的同义词替换
        expansions = [query]

        # 添加"是什么"变体
        if "?" in query or "？" in query:
            base = query.rstrip("?？")
            expansions.append(f"{base}是什么")
            expansions.append(f"解释{base}")

        return expansions[:3]

    def simplify_query(self, query: str) -> str:
        """
        简单简化: 去除常见停用词
        """
        # 去除一些停用词
        stopwords = ["请", "帮我", "我想", "能否", "可以"]
        simplified = query

        for word in stopwords:
            simplified = simplified.replace(word, "")

        return simplified.strip()
