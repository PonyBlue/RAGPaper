"""
日志系统模块
提供统一的日志管理，包括文件日志、控制台日志、性能监控等
"""

import logging
import os
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from datetime import datetime
import json


class RAGLogger:
    """RAG系统日志管理器"""

    def __init__(
        self,
        name: str = "rag_system",
        log_dir: str = "logs",
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        初始化日志器

        Args:
            name: 日志器名称
            log_dir: 日志目录
            level: 日志级别
            max_bytes: 单个日志文件最大大小
            backup_count: 保留的备份文件数量
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 避免重复添加handler
        if not self.logger.handlers:
            self._setup_handlers(max_bytes, backup_count)

        # 性能监控数据
        self.performance_data = []

    def _setup_handlers(self, max_bytes: int, backup_count: int):
        """设置日志处理器"""

        # 1. 控制台处理器（简洁格式）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # 2. 文件处理器（详细格式）
        file_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # 3. 错误日志单独文件
        error_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}_error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)

    def info(self, message: str, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(message, extra=kwargs)

    def log_query(
        self,
        query: str,
        response_time: float,
        num_results: int,
        strategy: str = "default"
    ):
        """
        记录查询日志

        Args:
            query: 查询内容
            response_time: 响应时间（秒）
            num_results: 返回结果数
            strategy: 检索策略
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # 限制长度
            "response_time": round(response_time, 3),
            "num_results": num_results,
            "strategy": strategy
        }

        self.logger.info(
            f"Query: '{query[:50]}...' | Time: {response_time:.3f}s | Results: {num_results} | Strategy: {strategy}"
        )

        # 保存到性能数据
        self.performance_data.append(log_entry)

        # 写入查询日志文件
        query_log_file = self.log_dir / "queries.jsonl"
        with open(query_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def log_retrieval(
        self,
        query: str,
        num_candidates: int,
        num_returned: int,
        retrieval_time: float,
        method: str = "hybrid"
    ):
        """
        记录检索日志

        Args:
            query: 查询内容
            num_candidates: 候选文档数
            num_returned: 返回文档数
            retrieval_time: 检索时间
            method: 检索方法
        """
        self.logger.info(
            f"Retrieval: {method} | Query: '{query[:50]}...' | "
            f"Candidates: {num_candidates} -> Returned: {num_returned} | "
            f"Time: {retrieval_time:.3f}s"
        )

    def log_generation(
        self,
        query: str,
        answer_length: int,
        generation_time: float,
        model: str = "gpt-3.5-turbo"
    ):
        """
        记录生成日志

        Args:
            query: 查询内容
            answer_length: 答案长度
            generation_time: 生成时间
            model: 模型名称
        """
        self.logger.info(
            f"Generation: {model} | Query: '{query[:50]}...' | "
            f"Answer Length: {answer_length} chars | "
            f"Time: {generation_time:.3f}s"
        )

    def log_performance(
        self,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        记录性能指标

        Args:
            operation: 操作名称
            duration: 耗时（秒）
            metadata: 额外元数据
        """
        perf_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": round(duration, 3),
            "metadata": metadata or {}
        }

        self.logger.debug(
            f"Performance: {operation} took {duration:.3f}s"
        )

        # 写入性能日志
        perf_log_file = self.log_dir / "performance.jsonl"
        with open(perf_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(perf_entry, ensure_ascii=False) + '\n')

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.performance_data:
            return {
                "total_queries": 0,
                "avg_response_time": 0.0,
                "min_response_time": 0.0,
                "max_response_time": 0.0
            }

        response_times = [d["response_time"] for d in self.performance_data]

        return {
            "total_queries": len(self.performance_data),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "recent_queries": self.performance_data[-10:]  # 最近10条
        }

    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any]
    ):
        """
        记录带上下文的错误

        Args:
            error: 异常对象
            context: 错误上下文信息
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }

        self.logger.error(
            f"Error: {type(error).__name__} - {str(error)} | Context: {context}"
        )

        # 写入错误日志
        error_log_file = self.log_dir / "errors.jsonl"
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_entry, ensure_ascii=False) + '\n')


class PerformanceTimer:
    """性能计时器上下文管理器"""

    def __init__(self, logger: RAGLogger, operation: str, metadata: Optional[Dict] = None):
        """
        初始化计时器

        Args:
            logger: 日志器实例
            operation: 操作名称
            metadata: 额外元数据
        """
        self.logger = logger
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None

    def __enter__(self):
        """开始计时"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """结束计时并记录"""
        duration = time.time() - self.start_time
        self.logger.log_performance(self.operation, duration, self.metadata)
        return False


# ==================== 全局日志器实例 ====================

# 主系统日志器
system_logger = RAGLogger(
    name="rag_system",
    log_dir="logs",
    level=logging.INFO
)

# 检索日志器
retrieval_logger = RAGLogger(
    name="retrieval",
    log_dir="logs",
    level=logging.DEBUG
)

# 生成日志器
generation_logger = RAGLogger(
    name="generation",
    log_dir="logs",
    level=logging.DEBUG
)


# ==================== 便捷函数 ====================

def get_logger(name: str = "rag_system") -> RAGLogger:
    """
    获取指定名称的日志器

    Args:
        name: 日志器名称

    Returns:
        RAGLogger实例
    """
    if name == "retrieval":
        return retrieval_logger
    elif name == "generation":
        return generation_logger
    else:
        return system_logger


def log_function_call(func):
    """
    装饰器：自动记录函数调用

    用法:
        @log_function_call
        def my_function(arg1, arg2):
            pass
    """
    def wrapper(*args, **kwargs):
        logger = system_logger
        func_name = func.__name__

        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")

        try:
            with PerformanceTimer(logger, func_name):
                result = func(*args, **kwargs)
            logger.debug(f"{func_name} completed successfully")
            return result

        except Exception as e:
            logger.log_error_with_context(
                error=e,
                context={
                    "function": func_name,
                    "args": str(args)[:100],
                    "kwargs": str(kwargs)[:100]
                }
            )
            raise

    return wrapper


if __name__ == "__main__":
    # 测试日志系统
    print("Testing RAG Logger...")

    # 基础日志
    system_logger.info("System initialized")
    system_logger.warning("This is a warning")
    system_logger.error("This is an error")

    # 查询日志
    system_logger.log_query(
        query="What is QMDF?",
        response_time=1.23,
        num_results=4,
        strategy="hybrid"
    )

    # 检索日志
    retrieval_logger.log_retrieval(
        query="What is QMDF?",
        num_candidates=20,
        num_returned=4,
        retrieval_time=0.5,
        method="hybrid"
    )

    # 生成日志
    generation_logger.log_generation(
        query="What is QMDF?",
        answer_length=256,
        generation_time=0.73,
        model="gpt-3.5-turbo"
    )

    # 性能计时器
    with PerformanceTimer(system_logger, "test_operation"):
        time.sleep(0.1)

    # 错误日志
    try:
        raise ValueError("Test error")
    except Exception as e:
        system_logger.log_error_with_context(
            error=e,
            context={"test": "error logging"}
        )

    # 性能统计
    stats = system_logger.get_performance_stats()
    print(f"\nPerformance Stats: {stats}")

    print("\n[OK] Logger test completed. Check logs/ directory.")
