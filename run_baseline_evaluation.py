"""
Baseline评估脚本
运行当前RAG系统的基准测试，记录初始性能指标
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evaluation import RAGEvaluator, TestCaseGenerator
from src.config import *
from src.document_loader import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.qa_chain import QAChainManager


def create_test_cases_for_papers():
    """
    为已上传的论文创建测试用例
    请根据实际上传的论文内容修改这些测试用例
    """
    print("📝 创建测试用例...")

    # 示例测试用例（需要根据实际论文内容修改）
    test_cases = [
        {
            "query": "论文的主要研究目标是什么？",
            "relevant_docs": ["paper_p0", "paper_p1"],  # 格式: 文件名_p页码
            "reference_answer": "论文的主要研究目标是...",
            "source": "example.pdf"
        },
        {
            "query": "使用了什么方法或算法？",
            "relevant_docs": ["paper_p2", "paper_p3"],
            "reference_answer": "使用的方法包括...",
            "source": "example.pdf"
        },
        {
            "query": "在哪些数据集上进行了实验？",
            "relevant_docs": ["paper_p5", "paper_p6"],
            "reference_answer": "实验使用了...数据集",
            "source": "example.pdf"
        },
        {
            "query": "实验结果表现如何？",
            "relevant_docs": ["paper_p7", "paper_p8"],
            "reference_answer": "实验结果显示...",
            "source": "example.pdf"
        },
        {
            "query": "这项工作有什么局限性？",
            "relevant_docs": ["paper_p9", "paper_p10"],
            "reference_answer": "主要局限性包括...",
            "source": "example.pdf"
        },
    ]

    # 保存测试用例
    test_case_file = "data/test_cases/baseline_test_cases.json"
    TestCaseGenerator.save_test_cases(test_cases, test_case_file)

    return test_cases


def check_system_ready():
    """检查系统是否准备好进行评估"""
    print("🔍 检查系统状态...")

    # 检查向量数据库是否存在
    db_path = Path(CHROMA_PERSIST_DIR)
    if not db_path.exists() or not list(db_path.glob("*")):
        print("⚠️  警告: 向量数据库为空！")
        print("   请先上传PDF论文并构建向量数据库。")
        print("   运行: streamlit run app.py")
        return False

    print("✅ 向量数据库已存在")
    return True


def run_baseline_evaluation():
    """运行baseline评估"""
    print("\n" + "=" * 70)
    print("🚀 开始Baseline评估")
    print("=" * 70)

    # 1. 检查系统状态
    if not check_system_ready():
        print("\n❌ 系统未准备好，请先上传论文并构建向量数据库")
        print("   运行命令: streamlit run app.py")
        return

    # 2. 创建测试用例
    print("\n" + "-" * 70)
    test_cases = create_test_cases_for_papers()
    print(f"✅ 已创建 {len(test_cases)} 个测试用例")

    # 3. 初始化RAG系统组件
    print("\n" + "-" * 70)
    print("🔧 初始化RAG系统...")

    try:
        # 初始化向量数据库
        print("  [1/3] 加载向量数据库...")
        vs_manager = VectorStoreManager(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_model_name=EMBEDDING_MODEL
        )
        vs_manager.load_vectorstore()

        # 初始化问答链
        print("  [2/3] 初始化问答链...")
        qa_manager = QAChainManager(
            model_name=LLM_MODEL,
            temperature=TEMPERATURE
        )

        # 创建检索器
        print("  [3/3] 创建检索器...")
        retriever = vs_manager.get_retriever(k=TOP_K)
        qa_chain = qa_manager.create_chain(retriever)

        print("✅ RAG系统初始化完成")

    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("   请确保已经上传论文并构建了向量数据库")
        return

    # 4. 创建评估器并运行评估
    print("\n" + "-" * 70)
    print("📊 开始评估...")

    evaluator = RAGEvaluator(output_dir="results")

    # 创建一个简单的RAG系统包装类
    class SimpleRAGSystem:
        def __init__(self, retriever, qa_chain):
            self.retriever = retriever
            self.qa_chain = qa_chain

    rag_system = SimpleRAGSystem(retriever, qa_chain)

    # 运行端到端评估
    try:
        results = evaluator.evaluate_end_to_end(
            test_cases=test_cases,
            rag_system=rag_system,
            k=TOP_K
        )

        print("\n" + "=" * 70)
        print("✅ Baseline评估完成！")
        print("=" * 70)
        print("\n💡 提示:")
        print("  - 评估结果已保存到 results/ 目录")
        print("  - 这些指标将作为后续优化的对比基准")
        print("  - 接下来可以开始实施优化方案（Rerank、混合检索等）")
        print("\n")

        return results

    except Exception as e:
        print(f"\n❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           RAG系统 Baseline 评估工具                           ║")
    print("║                                                                ║")
    print("║  本工具将评估当前RAG系统的性能，建立优化基准                  ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print("\n")

    # 运行评估
    results = run_baseline_evaluation()

    if results:
        print("🎯 下一步:")
        print("  1. 查看评估结果: results/evaluation_*.json")
        print("  2. 开始实施优化: 从Rerank重排序开始")
        print("  3. 优化后重新运行此脚本，对比效果")
        print("\n")


if __name__ == "__main__":
    main()
