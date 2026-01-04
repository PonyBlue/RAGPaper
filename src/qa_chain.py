from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.config import OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME


class QAChainManager:
    """问答链管理器"""

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        api_base: str = OPENAI_API_BASE,
        model_name: str = MODEL_NAME,
        temperature: float = 0.1
    ):
        # 初始化LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model=model_name,
            temperature=temperature
        )

        # 定义Prompt模板
        self.prompt_template = """你是一个专业的科研论文分析助手，专注于计算机图形学领域。
请基于以下提供的论文片段来回答问题。

重要要求：
1. 仅根据提供的上下文回答问题，不要编造信息
2. 如果上下文中没有相关信息，请明确说明"根据提供的文档，我无法回答这个问题"
3. 在回答时，请指出信息来源于哪篇论文（如果上下文中有提供）
4. 使用专业但易懂的语言

上下文信息：
{context}

问题：{question}

详细回答："""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def create_qa_chain(self, retriever):
        """创建问答链"""
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        return qa_chain

    def ask(self, qa_chain, question: str) -> Dict:
        """提问并获取答案"""
        result = qa_chain.invoke({"query": question})
        return result

    def format_response(self, result: Dict) -> str:
        """格式化响应结果，包含引文溯源"""
        answer = result["result"]
        source_documents = result.get("source_documents", [])

        formatted_response = f"**回答：**\n{answer}\n\n"

        if source_documents:
            formatted_response += "**参考来源：**\n"
            seen_sources = set()

            for i, doc in enumerate(source_documents, 1):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page", "未知页码")
                source_key = f"{source}-{page}"

                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    content_preview = doc.page_content[:150].replace("\n", " ")
                    formatted_response += f"\n{i}. **文档:** {source} (第{page}页)\n"
                    formatted_response += f"   **内容片段:** {content_preview}...\n"

        return formatted_response

    def format_response_with_scores(
        self,
        result: Dict,
        scores: List[float] = None
    ) -> str:
        """
        增强的格式化响应，包含相似度分数

        Args:
            result: QA chain返回的结果
            scores: 相似度分数列表（可选）

        Returns:
            格式化的响应文本
        """
        answer = result["result"]
        source_documents = result.get("source_documents", [])

        # 回答部分
        formatted_response = "=" * 70 + "\n"
        formatted_response += "【回答】\n"
        formatted_response += "=" * 70 + "\n"
        formatted_response += f"{answer}\n\n"

        # 参考来源部分
        if source_documents:
            formatted_response += "=" * 70 + "\n"
            formatted_response += "【参考来源与溯源信息】\n"
            formatted_response += "=" * 70 + "\n\n"

            seen_sources = set()

            for i, doc in enumerate(source_documents, 1):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page", "未知页码")
                source_key = f"{source}-{page}"

                if source_key not in seen_sources:
                    seen_sources.add(source_key)

                    # 引用标题
                    formatted_response += f"[引用 {i}]\n"
                    formatted_response += "-" * 70 + "\n"

                    # 文档信息
                    formatted_response += f"  文档来源: {source}\n"
                    formatted_response += f"  页码位置: 第 {page} 页\n"

                    # 相似度分数（如果提供）
                    if scores and i <= len(scores):
                        score = scores[i-1]
                        similarity_percent = score * 100
                        # 根据分数给出相关度评级
                        if similarity_percent >= 80:
                            relevance = "高度相关"
                        elif similarity_percent >= 60:
                            relevance = "较为相关"
                        elif similarity_percent >= 40:
                            relevance = "一般相关"
                        else:
                            relevance = "相关性较低"

                        formatted_response += f"  相似度评分: {similarity_percent:.1f}% ({relevance})\n"

                    # 内容片段
                    content_preview = doc.page_content[:300].replace("\n", " ").strip()
                    formatted_response += f"\n  【内容片段】\n"
                    formatted_response += f"  \"{content_preview}...\"\n\n"

            formatted_response += "=" * 70 + "\n"
            formatted_response += "【说明】以上引用按相关度排序，您可以根据页码查阅原文获取更多细节\n"
            formatted_response += "=" * 70 + "\n"

        return formatted_response

    def format_response_simple(self, result: Dict) -> str:
        """
        简洁版格式化响应（适合命令行显示）

        Args:
            result: QA chain返回的结果

        Returns:
            简洁格式的响应文本
        """
        answer = result["result"]
        source_documents = result.get("source_documents", [])

        formatted_response = f"\n【回答】\n{answer}\n"

        if source_documents:
            formatted_response += f"\n【来源】共 {len(source_documents)} 个相关文档片段\n"

            seen_sources = {}
            for doc in source_documents:
                source = doc.metadata.get("source", "未知")
                page = doc.metadata.get("page", "?")

                if source not in seen_sources:
                    seen_sources[source] = []
                if page not in seen_sources[source]:
                    seen_sources[source].append(page)

            for source, pages in seen_sources.items():
                pages_str = ", ".join([f"p{p}" for p in sorted(pages)])
                formatted_response += f"  - {source} ({pages_str})\n"

        return formatted_response

    def simple_ask(self, qa_chain, question: str) -> str:
        """简化的提问接口，直接返回格式化的答案"""
        result = self.ask(qa_chain, question)
        return self.format_response(result)
