"""
RAG系统评估模块
提供检索和生成质量的量化评估指标
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document


class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, output_dir: str = "results"):
        """
        初始化评估器

        Args:
            output_dir: 评估结果保存目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_history = []

    # ==================== 检索质量评估 ====================

    def evaluate_retrieval(
        self,
        test_cases: List[Dict],
        retriever_func,
        k: int = 4
    ) -> Dict[str, float]:
        """
        评估检索效果

        Args:
            test_cases: 测试用例列表，格式:
                [
                    {
                        "query": "问题",
                        "relevant_docs": ["相关文档1", "相关文档2"],  # 真实相关文档
                        "source": "来源论文名"
                    },
                    ...
                ]
            retriever_func: 检索函数，输入query，返回List[Document]
            k: Top-K

        Returns:
            metrics: 包含各项指标的字典
        """
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        ndcg_scores = []

        for case in test_cases:
            query = case["query"]
            relevant_docs = set(case.get("relevant_docs", []))

            # 执行检索
            retrieved_docs = retriever_func(query, k=k)
            retrieved_ids = [self._get_doc_id(doc) for doc in retrieved_docs]

            # 计算Precision@K
            precision = self._calculate_precision_at_k(retrieved_ids, relevant_docs, k)
            precision_scores.append(precision)

            # 计算Recall@K
            recall = self._calculate_recall_at_k(retrieved_ids, relevant_docs)
            recall_scores.append(recall)

            # 计算MRR (Mean Reciprocal Rank)
            mrr = self._calculate_mrr(retrieved_ids, relevant_docs)
            mrr_scores.append(mrr)

            # 计算NDCG (Normalized Discounted Cumulative Gain)
            ndcg = self._calculate_ndcg(retrieved_ids, relevant_docs, k)
            ndcg_scores.append(ndcg)

        metrics = {
            f"precision@{k}": np.mean(precision_scores),
            f"recall@{k}": np.mean(recall_scores),
            "mrr": np.mean(mrr_scores),
            f"ndcg@{k}": np.mean(ndcg_scores),
            "num_queries": len(test_cases)
        }

        return metrics

    def evaluate_retrieval_with_scores(
        self,
        test_cases: List[Dict],
        retriever_func_with_scores,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        评估检索效果（带相似度分数）

        Args:
            retriever_func_with_scores: 检索函数，返回List[Tuple[Document, float]]
        """
        metrics = self.evaluate_retrieval(
            test_cases,
            lambda q, k: [doc for doc, _ in retriever_func_with_scores(q, k)],
            k=k
        )

        # 额外统计分数分布
        all_scores = []
        for case in test_cases:
            query = case["query"]
            results = retriever_func_with_scores(query, k=k)
            scores = [score for _, score in results]
            all_scores.extend(scores)

        metrics["avg_similarity_score"] = np.mean(all_scores) if all_scores else 0
        metrics["min_similarity_score"] = np.min(all_scores) if all_scores else 0
        metrics["max_similarity_score"] = np.max(all_scores) if all_scores else 0

        return metrics

    def _calculate_precision_at_k(
        self,
        retrieved: List[str],
        relevant: set,
        k: int
    ) -> float:
        """计算Precision@K"""
        if not retrieved or not relevant:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)

        return relevant_retrieved / k

    def _calculate_recall_at_k(
        self,
        retrieved: List[str],
        relevant: set
    ) -> float:
        """计算Recall@K"""
        if not relevant:
            return 0.0

        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)

        return relevant_retrieved / len(relevant)

    def _calculate_mrr(
        self,
        retrieved: List[str],
        relevant: set
    ) -> float:
        """计算MRR (Mean Reciprocal Rank)"""
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    def _calculate_ndcg(
        self,
        retrieved: List[str],
        relevant: set,
        k: int
    ) -> float:
        """计算NDCG@K"""
        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(rank + 1)

        # IDCG (Ideal DCG)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))

        return dcg / idcg if idcg > 0 else 0.0

    def _get_doc_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        # 使用source + page作为文档ID
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        return f"{source}_p{page}"

    # ==================== 生成质量评估 ====================

    def evaluate_generation(
        self,
        test_cases: List[Dict],
        qa_func
    ) -> Dict[str, float]:
        """
        评估生成质量

        Args:
            test_cases: 测试用例，格式:
                [
                    {
                        "query": "问题",
                        "reference_answer": "参考答案"
                    },
                    ...
                ]
            qa_func: 问答函数，输入query，返回answer字符串

        Returns:
            metrics: 生成质量指标
        """
        # 简化版评估（不依赖外部库）
        answer_lengths = []
        response_qualities = []

        for case in test_cases:
            query = case["query"]
            reference = case.get("reference_answer", "")

            # 执行问答
            answer = qa_func(query)

            # 统计答案长度
            answer_lengths.append(len(answer))

            # 简单的质量评分（基于长度和关键词）
            quality = self._simple_quality_score(answer, reference)
            response_qualities.append(quality)

        metrics = {
            "avg_answer_length": np.mean(answer_lengths),
            "avg_quality_score": np.mean(response_qualities),
            "num_queries": len(test_cases)
        }

        return metrics

    def _simple_quality_score(self, answer: str, reference: str) -> float:
        """
        简单的质量评分
        基于答案长度和与参考答案的词汇重叠度
        """
        if not answer:
            return 0.0

        # 长度合理性（50-500字较好）
        length_score = 1.0 if 50 <= len(answer) <= 500 else 0.5

        # 词汇重叠度（如果有参考答案）
        overlap_score = 0.0
        if reference:
            answer_words = set(answer.split())
            reference_words = set(reference.split())
            if reference_words:
                overlap = len(answer_words & reference_words)
                overlap_score = overlap / len(reference_words)

        return (length_score + overlap_score) / 2

    # ==================== 端到端评估 ====================

    def evaluate_end_to_end(
        self,
        test_cases: List[Dict],
        rag_system,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        端到端评估整个RAG系统

        Args:
            test_cases: 完整的测试用例
            rag_system: RAG系统对象（需要有retriever和qa_chain）
            k: Top-K

        Returns:
            完整的评估报告
        """
        print("=" * 60)
        print("开始RAG系统端到端评估...")
        print("=" * 60)

        # 1. 检索评估
        print("\n[1/3] 评估检索质量...")
        retrieval_metrics = self.evaluate_retrieval(
            test_cases,
            lambda q, k: rag_system.retriever.get_relevant_documents(q)[:k],
            k=k
        )

        # 2. 生成评估
        print("\n[2/3] 评估生成质量...")
        generation_metrics = self.evaluate_generation(
            test_cases,
            lambda q: rag_system.qa_chain.invoke({"query": q})["result"]
        )

        # 3. 性能指标
        print("\n[3/3] 评估系统性能...")
        performance_metrics = self._evaluate_performance(test_cases, rag_system)

        # 汇总结果
        full_report = {
            "timestamp": datetime.now().isoformat(),
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "performance_metrics": performance_metrics,
            "test_cases_count": len(test_cases)
        }

        # 保存结果
        self._save_results(full_report)

        # 打印报告
        self._print_report(full_report)

        return full_report

    def _evaluate_performance(
        self,
        test_cases: List[Dict],
        rag_system
    ) -> Dict[str, float]:
        """评估系统性能（响应时间等）"""
        import time

        retrieval_times = []
        generation_times = []
        total_times = []

        for case in test_cases[:5]:  # 只测试前5个用例
            query = case["query"]

            # 测量检索时间
            start = time.time()
            rag_system.retriever.get_relevant_documents(query)
            retrieval_time = time.time() - start
            retrieval_times.append(retrieval_time)

            # 测量总时间
            start = time.time()
            rag_system.qa_chain.invoke({"query": query})
            total_time = time.time() - start
            total_times.append(total_time)

            # 生成时间 = 总时间 - 检索时间
            generation_times.append(total_time - retrieval_time)

        return {
            "avg_retrieval_time": np.mean(retrieval_times),
            "avg_generation_time": np.mean(generation_times),
            "avg_total_time": np.mean(total_times),
            "max_total_time": np.max(total_times)
        }

    # ==================== 结果管理 ====================

    def _save_results(self, report: Dict):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"evaluation_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n[OK] 评估结果已保存到: {output_file}")

        # 添加到历史
        self.results_history.append(report)

    def _print_report(self, report: Dict):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("[评估报告] RAG系统评估报告")
        print("=" * 60)

        # 检索指标
        print("\n【检索质量】")
        retrieval = report["retrieval_metrics"]
        for key, value in retrieval.items():
            if key != "num_queries":
                print(f"  {key}: {value:.4f} ({value*100:.2f}%)")

        # 生成指标
        print("\n【生成质量】")
        generation = report["generation_metrics"]
        for key, value in generation.items():
            if key != "num_queries":
                print(f"  {key}: {value:.2f}")

        # 性能指标
        print("\n【系统性能】")
        performance = report["performance_metrics"]
        for key, value in performance.items():
            print(f"  {key}: {value:.3f}秒")

        print("\n" + "=" * 60)

    def compare_results(self, baseline_file: str, current_file: str):
        """对比两次评估结果"""
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline = json.load(f)

        with open(current_file, 'r', encoding='utf-8') as f:
            current = json.load(f)

        print("\n" + "=" * 60)
        print("[对比报告] 优化效果对比")
        print("=" * 60)

        # 对比检索指标
        print("\n【检索质量对比】")
        for key in baseline["retrieval_metrics"]:
            if key != "num_queries":
                base_val = baseline["retrieval_metrics"][key]
                curr_val = current["retrieval_metrics"][key]
                improvement = ((curr_val - base_val) / base_val * 100) if base_val > 0 else 0
                arrow = "▲" if improvement > 0 else "▼"
                print(f"  {key}:")
                print(f"    Baseline: {base_val:.4f}")
                print(f"    Current:  {curr_val:.4f}")
                print(f"    {arrow} 变化: {improvement:+.2f}%")

        # 对比性能指标
        print("\n【性能对比】")
        for key in baseline["performance_metrics"]:
            base_val = baseline["performance_metrics"][key]
            curr_val = current["performance_metrics"][key]
            improvement = ((base_val - curr_val) / base_val * 100) if base_val > 0 else 0
            arrow = "▲" if improvement > 0 else "▼"
            print(f"  {key}:")
            print(f"    Baseline: {base_val:.3f}秒")
            print(f"    Current:  {curr_val:.3f}秒")
            print(f"    {arrow} 变化: {improvement:+.2f}%")

        print("\n" + "=" * 60)


# ==================== 测试数据生成工具 ====================

class TestCaseGenerator:
    """测试用例生成器"""

    @staticmethod
    def generate_sample_test_cases() -> List[Dict]:
        """
        生成示例测试用例
        实际使用时应该根据具体的论文内容创建
        """
        test_cases = [
            {
                "query": "这篇论文的主要贡献是什么？",
                "relevant_docs": ["paper_p1", "paper_p2"],
                "reference_answer": "论文的主要贡献包括...",
                "source": "paper.pdf"
            },
            {
                "query": "使用了什么数据集？",
                "relevant_docs": ["paper_p5", "paper_p6"],
                "reference_answer": "使用了XXX数据集...",
                "source": "paper.pdf"
            },
            {
                "query": "实验结果如何？",
                "relevant_docs": ["paper_p8", "paper_p9"],
                "reference_answer": "实验结果表明...",
                "source": "paper.pdf"
            }
        ]

        return test_cases

    @staticmethod
    def save_test_cases(test_cases: List[Dict], output_path: str):
        """保存测试用例到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, ensure_ascii=False, indent=2)
        print(f"[OK] 测试用例已保存到: {output_path}")

    @staticmethod
    def load_test_cases(input_path: str) -> List[Dict]:
        """从JSON文件加载测试用例"""
        with open(input_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        print(f"[OK] 已加载 {len(test_cases)} 个测试用例")
        return test_cases
