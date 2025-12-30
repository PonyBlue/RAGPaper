import streamlit as st
import os
from pathlib import Path
from src.document_loader import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.qa_chain import QAChainManager
from src.config import DATA_DIR, CHROMA_DB_DIR

# 页面配置
st.set_page_config(
    page_title="科研论文智能分析助手",
    page_icon="📚",
    layout="wide"
)

# 初始化session state
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def init_system():
    """初始化系统组件"""
    try:
        # 初始化向量数据库管理器
        vs_manager = VectorStoreManager()

        # 尝试加载已有的向量数据库
        vectorstore = vs_manager.load_vectorstore()

        if vectorstore is None:
            st.warning("未找到已有的向量数据库，请先上传PDF文档建立知识库")
            return None, vs_manager

        # 初始化问答链
        qa_manager = QAChainManager()
        retriever = vs_manager.get_retriever()
        qa_chain = qa_manager.create_qa_chain(retriever)

        return qa_chain, vs_manager

    except Exception as e:
        st.error(f"初始化系统失败: {str(e)}")
        return None, None


def process_uploaded_files(uploaded_files, vs_manager):
    """处理上传的PDF文件"""
    if not uploaded_files:
        return

    with st.spinner("正在处理上传的文档..."):
        # 保存上传的文件
        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = DATA_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(str(file_path))
            st.success(f"已保存: {uploaded_file.name}")

        # 处理PDF文档
        doc_processor = DocumentProcessor()
        all_chunks = doc_processor.process_multiple_pdfs(saved_paths)

        # 创建或更新向量数据库
        if vs_manager.vectorstore is None:
            vs_manager.create_vectorstore(all_chunks)
            st.success("向量数据库创建成功！")
        else:
            vs_manager.add_documents(all_chunks)
            st.success("文档已添加到现有知识库！")

        st.session_state.vectorstore_loaded = True


def main():
    st.title("📚 科研论文智能分析助手")
    st.markdown("---")

    # 侧边栏：文档管理
    with st.sidebar:
        st.header("📂 文档管理")

        # 上传PDF
        uploaded_files = st.file_uploader(
            "上传PDF论文",
            type=["pdf"],
            accept_multiple_files=True,
            help="支持批量上传PDF格式的科研论文"
        )

        if st.button("处理上传的文档", type="primary"):
            if uploaded_files:
                _, vs_manager = init_system()
                if vs_manager is None:
                    vs_manager = VectorStoreManager()
                process_uploaded_files(uploaded_files, vs_manager)
                st.rerun()
            else:
                st.warning("请先上传PDF文件")

        st.markdown("---")

        # 显示已有文档
        st.subheader("已上传的文档")
        pdf_files = list(DATA_DIR.glob("*.pdf"))
        if pdf_files:
            for pdf_file in pdf_files:
                st.text(f"📄 {pdf_file.name}")
        else:
            st.info("暂无文档")

        st.markdown("---")

        # 系统设置
        st.subheader("⚙️ 系统设置")

        # 清理选项
        clean_option = st.selectbox(
            "清理数据",
            ["不清理", "仅清理向量库", "仅清理PDF文件", "清理所有数据"],
            help="选择要清理的数据类型"
        )

        if clean_option != "不清理":
            if st.button(f"确认{clean_option}", type="secondary"):
                if st.session_state.get("confirm_clean"):
                    import shutil

                    if clean_option == "仅清理向量库":
                        vs_manager = VectorStoreManager()
                        vs_manager.load_vectorstore()
                        vs_manager.delete_collection()
                        st.session_state.vectorstore_loaded = False
                        st.session_state.qa_chain = None
                        st.success("✓ 向量库已清理")

                    elif clean_option == "仅清理PDF文件":
                        for pdf in DATA_DIR.glob("*.pdf"):
                            pdf.unlink()
                        st.success("✓ PDF文件已清理")
                        st.warning("⚠️ 向量库中仍保留旧数据，建议同时清理向量库")

                    elif clean_option == "清理所有数据":
                        # 清理PDF
                        for pdf in DATA_DIR.glob("*.pdf"):
                            pdf.unlink()
                        # 清理向量库
                        if CHROMA_DB_DIR.exists():
                            shutil.rmtree(CHROMA_DB_DIR)
                            CHROMA_DB_DIR.mkdir()
                        st.session_state.vectorstore_loaded = False
                        st.session_state.qa_chain = None
                        st.success("✓ 所有数据已清理")

                    st.session_state.confirm_clean = False
                    st.rerun()
                else:
                    st.session_state.confirm_clean = True
                    st.warning("⚠️ 再次点击确认清理")

    # 主区域：问答界面
    st.header("💬 智能问答")

    # API配置检查
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ 未检测到OpenAI API Key，请在环境变量中设置 OPENAI_API_KEY")
        st.info("您可以使用 .env 文件或直接设置环境变量")
        api_key_input = st.text_input("或在此输入API Key:", type="password")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

    # 初始化QA系统
    if st.session_state.qa_chain is None:
        qa_chain, _ = init_system()
        if qa_chain:
            st.session_state.qa_chain = qa_chain
            st.session_state.vectorstore_loaded = True

    # 显示聊天历史
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if st.session_state.vectorstore_loaded and st.session_state.qa_chain:
        if question := st.chat_input("请输入您的问题..."):
            # 显示用户问题
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.chat_history.append({"role": "user", "content": question})

            # 获取回答
            with st.chat_message("assistant"):
                with st.spinner("正在思考..."):
                    try:
                        qa_manager = QAChainManager()
                        result = qa_manager.ask(st.session_state.qa_chain, question)
                        formatted_answer = qa_manager.format_response(result)
                        st.markdown(formatted_answer)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": formatted_answer
                        })
                    except Exception as e:
                        error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    else:
        st.info("👆 请先在左侧上传PDF文档以建立知识库")

    # 清除对话历史
    if st.button("清除对话历史"):
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    main()
