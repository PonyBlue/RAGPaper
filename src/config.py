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

# Embedding模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# 文本分块配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 检索配置
TOP_K = 4  # 返回最相关的K个文档片段

# Chroma集合名称
COLLECTION_NAME = "research_papers"
