# -*- coding: utf-8 -*-
"""增强的检索器，特别处理元数据类问题"""
from typing import List
from langchain_core.documents import Document


class EnhancedRetriever:
    """增强检索器，支持混合检索策略"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def is_metadata_question(self, query: str) -> bool:
        """判断是否是元数据类问题"""
        metadata_keywords = [
            "作者", "author", "标题", "title", "摘要", "abstract",
            "发表", "published", "期刊", "journal", "会议", "conference",
            "年份", "year", "关键词", "keywords"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in metadata_keywords)

    def retrieve_for_metadata(self, query: str, k: int = 4) -> List[Document]:
        """针对元数据问题的检索策略"""
        # 获取更多候选（增加召回率）
        candidates = self.vectorstore.similarity_search(query, k=k*3)

        # 优先返回前几页的内容
        front_pages = []
        other_pages = []

        for doc in candidates:
            page_num = doc.metadata.get("page", 999)
            if page_num <= 3:  # 前3页
                front_pages.append(doc)
            else:
                other_pages.append(doc)

        # 前3页优先，然后是其他页
        result = front_pages[:k] if len(front_pages) >= k else front_pages + other_pages[:k-len(front_pages)]

        return result

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """智能检索：根据问题类型选择策略"""
        if self.is_metadata_question(query):
            return self.retrieve_for_metadata(query, k)
        else:
            return self.vectorstore.similarity_search(query, k=k)
