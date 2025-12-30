from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """PDF文档处理器：负责加载和分块"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """加载单个PDF文件"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 添加元数据
        for doc in documents:
            doc.metadata["source"] = Path(pdf_path).name

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为小块"""
        chunks = self.text_splitter.split_documents(documents)

        # 清理和验证chunks
        cleaned_chunks = []
        for chunk in chunks:
            # 确保page_content是字符串且非空
            if chunk.page_content and isinstance(chunk.page_content, str):
                # 清理文本：移除特殊字符和多余空白
                cleaned_text = chunk.page_content.strip()
                if cleaned_text:  # 确保清理后不是空字符串
                    chunk.page_content = cleaned_text
                    cleaned_chunks.append(chunk)

        return cleaned_chunks

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """处理PDF：加载并分块"""
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)

        print(f"已加载文档: {Path(pdf_path).name}")
        print(f"文档页数: {len(documents)}")
        print(f"分块数量: {len(chunks)}")

        return chunks

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """批量处理多个PDF文件"""
        all_chunks = []

        for pdf_path in pdf_paths:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"处理文件 {pdf_path} 时出错: {str(e)}")

        print(f"\n总计处理了 {len(pdf_paths)} 个文件，生成 {len(all_chunks)} 个文档块")
        return all_chunks
