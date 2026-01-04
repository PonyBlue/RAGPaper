"""
缓存系统模块
提供Embedding缓存和查询结果缓存，提升系统性能
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta


class LRUCache:
    """
    LRU (Least Recently Used) 缓存
    自动淘汰最久未使用的项
    """

    def __init__(self, capacity: int = 1000):
        """
        初始化LRU缓存

        Args:
            capacity: 缓存容量
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在返回None
        """
        if key not in self.cache:
            self.misses += 1
            return None

        # 移到末尾（最近使用）
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]

    def put(self, key: str, value: Any):
        """
        设置缓存项

        Args:
            key: 缓存键
            value: 缓存值
        """
        if key in self.cache:
            # 已存在，移到末尾
            self.cache.move_to_end(key)
        else:
            # 新增
            if len(self.cache) >= self.capacity:
                # 删除最旧的项（第一个）
                self.cache.popitem(last=False)

        self.cache[key] = value

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计

        Returns:
            统计信息字典
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0

        return {
            "capacity": self.capacity,
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2)
        }


class TTLCache:
    """
    带过期时间的缓存 (Time To Live)
    自动清理过期项
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        初始化TTL缓存

        Args:
            ttl_seconds: 过期时间（秒）
            max_size: 最大缓存数量
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在或过期返回None
        """
        if key not in self.cache:
            self.misses += 1
            return None

        value, expire_time = self.cache[key]

        # 检查是否过期
        if time.time() > expire_time:
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return value

    def put(self, key: str, value: Any):
        """
        设置缓存项

        Args:
            key: 缓存键
            value: 缓存值
        """
        # 清理过期项
        self._clean_expired()

        # 如果达到最大容量，删除最早的项
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        expire_time = time.time() + self.ttl_seconds
        self.cache[key] = (value, expire_time)

    def _clean_expired(self):
        """清理过期项"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, exp) in self.cache.items()
            if current_time > exp
        ]
        for k in expired_keys:
            del self.cache[k]

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        self._clean_expired()
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0

        return {
            "max_size": self.max_size,
            "size": len(self.cache),
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2)
        }


class EmbeddingCache:
    """
    Embedding向量缓存
    缓存文本的向量表示，避免重复计算
    """

    def __init__(
        self,
        capacity: int = 10000,
        persist_path: Optional[str] = None
    ):
        """
        初始化Embedding缓存

        Args:
            capacity: 缓存容量
            persist_path: 持久化路径（可选）
        """
        self.cache = LRUCache(capacity=capacity)
        self.persist_path = Path(persist_path) if persist_path else None

        # 如果有持久化路径，尝试加载
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

    def _hash_text(self, text: str) -> str:
        """文本哈希（作为缓存键）"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        获取文本的Embedding向量

        Args:
            text: 输入文本

        Returns:
            向量列表，如果不存在返回None
        """
        key = self._hash_text(text)
        return self.cache.get(key)

    def put(self, text: str, embedding: List[float]):
        """
        缓存文本的Embedding向量

        Args:
            text: 输入文本
            embedding: 向量表示
        """
        key = self._hash_text(text)
        self.cache.put(key, embedding)

    def get_batch(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[str]]:
        """
        批量获取Embedding

        Args:
            texts: 文本列表

        Returns:
            (embeddings, missing_texts)
            embeddings: 向量列表（未命中的为None）
            missing_texts: 未命中的文本列表
        """
        embeddings = []
        missing_texts = []

        for text in texts:
            emb = self.get(text)
            embeddings.append(emb)
            if emb is None:
                missing_texts.append(text)

        return embeddings, missing_texts

    def put_batch(self, texts: List[str], embeddings: List[List[float]]):
        """
        批量缓存Embedding

        Args:
            texts: 文本列表
            embeddings: 向量列表
        """
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding)

    def save_to_disk(self):
        """持久化到磁盘"""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.persist_path, 'wb') as f:
            pickle.dump(self.cache.cache, f)

        print(f"[Cache] Embedding cache saved to {self.persist_path}")

    def _load_from_disk(self):
        """从磁盘加载"""
        try:
            with open(self.persist_path, 'rb') as f:
                loaded_cache = pickle.load(f)
                self.cache.cache = loaded_cache

            print(f"[Cache] Loaded {len(self.cache.cache)} embeddings from {self.persist_path}")

        except Exception as e:
            print(f"[Warning] Failed to load embedding cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()


class QueryResultCache:
    """
    查询结果缓存
    缓存检索和生成的结果
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,  # 1小时
        max_size: int = 1000
    ):
        """
        初始化查询结果缓存

        Args:
            ttl_seconds: 过期时间
            max_size: 最大缓存数量
        """
        self.cache = TTLCache(ttl_seconds=ttl_seconds, max_size=max_size)

    def _make_key(
        self,
        query: str,
        k: int,
        strategy: str,
        **kwargs
    ) -> str:
        """
        生成缓存键

        Args:
            query: 查询文本
            k: 返回文档数
            strategy: 检索策略
            **kwargs: 其他参数

        Returns:
            缓存键
        """
        # 构建唯一键
        key_parts = [
            query,
            str(k),
            strategy,
            json.dumps(kwargs, sort_keys=True)
        ]
        key_string = "|".join(key_parts)

        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    def get(
        self,
        query: str,
        k: int = 4,
        strategy: str = "default",
        **kwargs
    ) -> Optional[Any]:
        """
        获取查询结果

        Args:
            query: 查询文本
            k: 返回文档数
            strategy: 检索策略
            **kwargs: 其他参数

        Returns:
            缓存的结果，如果不存在返回None
        """
        key = self._make_key(query, k, strategy, **kwargs)
        return self.cache.get(key)

    def put(
        self,
        query: str,
        result: Any,
        k: int = 4,
        strategy: str = "default",
        **kwargs
    ):
        """
        缓存查询结果

        Args:
            query: 查询文本
            result: 查询结果
            k: 返回文档数
            strategy: 检索策略
            **kwargs: 其他参数
        """
        key = self._make_key(query, k, strategy, **kwargs)
        self.cache.put(key, result)

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()


class CacheManager:
    """
    缓存管理器
    统一管理所有缓存
    """

    def __init__(
        self,
        enable_embedding_cache: bool = True,
        enable_query_cache: bool = True,
        embedding_capacity: int = 10000,
        query_ttl: int = 3600,
        persist_dir: Optional[str] = None
    ):
        """
        初始化缓存管理器

        Args:
            enable_embedding_cache: 启用Embedding缓存
            enable_query_cache: 启用查询缓存
            embedding_capacity: Embedding缓存容量
            query_ttl: 查询缓存过期时间
            persist_dir: 持久化目录
        """
        self.enable_embedding_cache = enable_embedding_cache
        self.enable_query_cache = enable_query_cache

        # 初始化缓存
        if enable_embedding_cache:
            persist_path = Path(persist_dir) / "embedding_cache.pkl" if persist_dir else None
            self.embedding_cache = EmbeddingCache(
                capacity=embedding_capacity,
                persist_path=persist_path
            )
        else:
            self.embedding_cache = None

        if enable_query_cache:
            self.query_cache = QueryResultCache(ttl_seconds=query_ttl)
        else:
            self.query_cache = None

        print(f"[Cache] Initialized - Embedding: {enable_embedding_cache}, Query: {enable_query_cache}")

    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有缓存统计"""
        stats = {
            "enabled": {
                "embedding_cache": self.enable_embedding_cache,
                "query_cache": self.enable_query_cache
            }
        }

        if self.embedding_cache:
            stats["embedding_cache"] = self.embedding_cache.get_stats()

        if self.query_cache:
            stats["query_cache"] = self.query_cache.get_stats()

        return stats

    def save_caches(self):
        """保存所有可持久化的缓存"""
        if self.embedding_cache:
            self.embedding_cache.save_to_disk()

    def clear_all(self):
        """清空所有缓存"""
        if self.embedding_cache:
            self.embedding_cache.cache.clear()
            print("[Cache] Embedding cache cleared")

        if self.query_cache:
            self.query_cache.cache.clear()
            print("[Cache] Query cache cleared")


# ==================== 全局缓存管理器 ====================

cache_manager = CacheManager(
    enable_embedding_cache=True,
    enable_query_cache=True,
    embedding_capacity=10000,
    query_ttl=3600,
    persist_dir="cache"
)


if __name__ == "__main__":
    # 测试缓存系统
    print("Testing Cache System...\n")

    # 测试LRU缓存
    print("=== LRU Cache Test ===")
    lru = LRUCache(capacity=3)

    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)
    print(f"After adding a,b,c: {lru.get_stats()}")

    lru.get("a")  # hit
    lru.get("d")  # miss
    print(f"After get(a), get(d): {lru.get_stats()}")

    lru.put("d", 4)  # 淘汰b
    print(f"After adding d: {lru.get_stats()}")

    # 测试TTL缓存
    print("\n=== TTL Cache Test ===")
    ttl = TTLCache(ttl_seconds=2, max_size=100)

    ttl.put("key1", "value1")
    print(f"After put: {ttl.get('key1')}")

    time.sleep(1)
    print(f"After 1s: {ttl.get('key1')}")

    time.sleep(1.5)
    print(f"After 2.5s (expired): {ttl.get('key1')}")

    print(f"TTL Stats: {ttl.get_stats()}")

    # 测试Embedding缓存
    print("\n=== Embedding Cache Test ===")
    emb_cache = EmbeddingCache(capacity=1000, persist_path="cache/test_emb.pkl")

    emb_cache.put("hello world", [0.1, 0.2, 0.3])
    result = emb_cache.get("hello world")
    print(f"Get embedding: {result}")

    # 批量测试
    texts = ["text1", "text2", "text3"]
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    emb_cache.put_batch(texts, embeddings)

    cached, missing = emb_cache.get_batch(texts + ["text4"])
    print(f"Batch get - Cached: {len([e for e in cached if e is not None])}, Missing: {len(missing)}")

    print(f"Embedding Cache Stats: {emb_cache.get_stats()}")

    # 测试查询结果缓存
    print("\n=== Query Result Cache Test ===")
    query_cache = QueryResultCache(ttl_seconds=60)

    query_cache.put("what is QMDF?", {"answer": "...", "docs": []}, k=4, strategy="hybrid")
    result = query_cache.get("what is QMDF?", k=4, strategy="hybrid")
    print(f"Get query result: {result is not None}")

    print(f"Query Cache Stats: {query_cache.get_stats()}")

    # 测试缓存管理器
    print("\n=== Cache Manager Test ===")
    manager = CacheManager(persist_dir="cache")
    stats = manager.get_all_stats()
    print(f"All stats: {json.dumps(stats, indent=2)}")

    print("\n[OK] Cache system test completed")
