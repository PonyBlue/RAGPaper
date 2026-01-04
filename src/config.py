import os
from pathlib import Path
from dotenv import load_dotenv

# 项目路径配置
BASE_DIR = Path(__file__).parent.parent

# 加载.env文件，强制覆盖已存在的环境变量
load_dotenv(BASE_DIR / ".env", override=True)
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# LLM配置（使用环境变量）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
LLM_MODEL = os.getenv("LLM_MODEL", MODEL_NAME)  # 兼容性别名
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))  # LLM温度参数

# Chroma持久化目录
CHROMA_PERSIST_DIR = str(CHROMA_DB_DIR)

# Embedding模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# 文本分块配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 检索配置
TOP_K = 4  # 返回最相关的K个文档片段

# Rerank配置
USE_RERANK = os.getenv("USE_RERANK", "false").lower() == "true"  # 是否启用Rerank
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")  # Rerank模型名称
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))  # Rerank后返回的文档数
FETCH_K = int(os.getenv("FETCH_K", "20"))  # 初步检索的文档数（用于Rerank）

# 混合检索配置
USE_HYBRID = os.getenv("USE_HYBRID", "false").lower() == "true"  # 是否启用混合检索
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))  # 语义检索权重（0-1）
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))  # BM25参数k1
BM25_B = float(os.getenv("BM25_B", "0.75"))  # BM25参数b

# 查询改写配置
USE_QUERY_REWRITE = os.getenv("USE_QUERY_REWRITE", "false").lower() == "true"  # 是否启用查询改写
REWRITE_STRATEGY = os.getenv("REWRITE_STRATEGY", "decompose")  # 改写策略: decompose, expand, both
MERGE_STRATEGY = os.getenv("MERGE_STRATEGY", "union")  # 合并策略: union, intersection, avg_score

# Chroma集合名称
COLLECTION_NAME = "research_papers"
