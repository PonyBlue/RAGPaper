# 📚 基于 RAG 的科研论文智能分析助手

> 一个专注于计算机图形学领域的智能论文问答系统，基于检索增强生成（RAG）技术实现

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 项目简介

本项目是一个基于检索增强生成（RAG）技术的科研论文智能分析系统，通过结合 LangChain、Chroma 向量数据库和 Streamlit，实现了高效的论文检索和智能问答功能。

### 核心特性

- 🔍 **智能检索**：基于语义相似度的文档检索，精准定位相关内容
- 💡 **引文溯源**：每个答案都标注来源，可追溯到原文档段落
- 📄 **PDF 处理**：自动解析 PDF 论文，支持批量上传
- 💾 **持久化存储**：向量数据库持久化，无需重复处理
- 🎨 **友好界面**：基于 Streamlit 的直观 Web 界面
- 🔧 **高度可配置**：支持自定义 API、模型参数等

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    用户界面 (Streamlit)               │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│              RAG 问答系统 (LangChain)                │
│  ┌──────────────┐    ┌──────────────┐              │
│  │  文档检索器   │ →  │   问答链      │              │
│  └──────────────┘    └──────────────┘              │
└─────────────────────────────────────────────────────┘
            ↓                          ↓
┌─────────────────────┐    ┌─────────────────────┐
│  向量数据库 (Chroma) │    │   LLM (OpenAI API)  │
│  - 文档向量          │    │   - gpt-3.5-turbo   │
│  - 持久化存储        │    │   - 自定义 API      │
└─────────────────────┘    └─────────────────────┘
            ↑
┌─────────────────────┐
│  PDF 文档处理        │
│  - PyPDF 解析        │
│  - 智能分块          │
│  - Embedding 向量化  │
└─────────────────────┘
```

### 技术栈

| 组件 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **框架** | LangChain | 1.0.0 | 构建 RAG 应用的主框架 |
| **向量数据库** | ChromaDB | 1.4.0 | 存储和检索文档向量 |
| **Embedding** | Sentence Transformers | 3.3.1 | 文本向量化模型 |
| **前端** | Streamlit | 1.52.2 | Web 界面框架 |
| **PDF 解析** | PyPDF | 6.5.0 | PDF 文档处理 |
| **LLM** | OpenAI API | - | 大语言模型接口 |

## 📁 项目结构

```
langchain/
├── 📂 data/                    # 存放上传的 PDF 文档
│   └── QMDF.pdf               # 示例论文
├── 📂 chroma_db/              # Chroma 向量数据库存储目录
├── 📂 src/                    # 源代码目录
│   ├── __init__.py
│   ├── config.py              # 配置参数
│   ├── document_loader.py     # PDF 解析和文档分块
│   ├── vectorstore.py         # 向量数据库管理
│   ├── qa_chain.py            # 问答链逻辑
│   └── enhanced_retriever.py  # 增强检索器
├── 📄 app.py                  # Streamlit 主应用
├── 📄 requirements.txt        # 项目依赖
├── 📄 .env                    # 环境变量配置（需自行创建）
├── 📄 cleanup.py              # 数据清理脚本
├── 📄 quick_test.py           # 快速验证脚本
├── 📄 start.bat               # Windows 启动脚本
├── 📄 fix_dependencies.bat    # 依赖修复脚本
├── 📄 README.md               # 项目说明文档
├── 📄 QUICKSTART.md           # 快速开始指南
├── 📄 DATA_MANAGEMENT.md      # 数据管理指南
├── 📄 OPTIMIZATION_PLAN.md    # 优化计划
└── 📄 BUGFIX_README.md        # 问题修复说明
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Conda（推荐）或 pip
- OpenAI API Key

### 1️⃣ 创建虚拟环境

```bash
# 使用 Conda（推荐）
conda create -n rag-paper python=3.10 -y
conda activate rag-paper

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

> ⚠️ **重要**：本项目使用特定版本组合以确保兼容性。如遇到问题，请参考 [BUGFIX_README.md](BUGFIX_README.md)

### 3️⃣ 配置 API Key

创建 `.env` 文件：

```bash
# .env 文件内容
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1  # 可选，自定义 API 地址
MODEL_NAME=gpt-3.5-turbo  # 可选，默认为 gpt-3.5-turbo
```

或设置环境变量：

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"
```

### 4️⃣ 启动应用

```bash
conda activate rag-paper
streamlit run app.py
```

或使用启动脚本（Windows）：

```bash
start.bat
```

应用将在浏览器中自动打开：`http://localhost:8501`

## 💻 使用指南

### 上传并处理文档

1. 在左侧边栏点击 **"浏览文件"** 上传 PDF 论文
2. 支持批量上传多个 PDF 文件
3. 点击 **"处理上传的文档"** 按钮
4. 系统会自动：
   - 解析 PDF 内容
   - 将文本分块
   - 生成向量 Embedding
   - 存储到向量数据库

### 智能问答

在主界面的输入框中输入问题，例如：

- "这篇论文的主要贡献是什么？"
- "作者使用了什么算法来解决该问题？"
- "实验结果如何？有哪些性能指标？"
- "这个方法与现有方法相比有什么优势？"
- "论文中提到的 Q-MDF 是什么？"

系统会：
1. 从向量数据库中检索最相关的文档片段
2. 将问题和检索结果发送给 LLM
3. 生成专业的答案
4. 显示答案来源（引文溯源）

### 数据管理

#### 方法 1：界面清理（推荐）

1. 左侧边栏 → **系统设置** → **清理数据**
2. 选择清理选项：
   - **仅清理向量库**：删除向量数据库，保留 PDF 文件
   - **仅清理 PDF 文件**：删除上传的 PDF，保留向量库
   - **清理所有数据**：删除所有数据
3. 点击确认按钮两次

#### 方法 2：使用清理脚本

```bash
python cleanup.py
```

提供交互式菜单：
- 查看系统状态
- 选择性清理
- 安全确认机制

#### 方法 3：手动清理

```bash
# 删除所有数据
rm -rf data/* chroma_db/*

# Windows
rmdir /s /q data chroma_db
mkdir data chroma_db
```

详见 [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md)

## ⚙️ 配置说明

### 核心配置 (`src/config.py`)

```python
# 文本分块配置
CHUNK_SIZE = 1000           # 每个文本块的字符数
CHUNK_OVERLAP = 200         # 文本块之间的重叠字符数

# 检索配置
TOP_K = 4                   # 返回最相关的 K 个文档片段

# Embedding 模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 向量数据库
CHROMA_DB_DIR = Path("chroma_db")
COLLECTION_NAME = "papers"

# LLM 配置
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.3
```

### 自定义 API Base

如果使用自定义 API（如国内镜像、本地部署等）：

```python
# .env 文件
OPENAI_API_BASE=https://your-custom-api.com/v1
```

## 🔧 故障排除

### 问题 1：TypeError: TextEncodeInput must be Union...

**原因**：库版本不兼容

**解决方案**：

```bash
conda activate rag-paper
pip uninstall -y sentence-transformers transformers tokenizers
pip install sentence-transformers==3.3.1 transformers==4.45.2 tokenizers==0.20.0
```

或运行修复脚本：
```bash
fix_dependencies.bat  # Windows
```

详见 [BUGFIX_README.md](BUGFIX_README.md)

### 问题 2：找不到 API Key

**解决方案**：

1. 检查 `.env` 文件是否存在且格式正确
2. 确认环境变量已设置
3. 在应用界面直接输入 API Key

### 问题 3：PDF 处理失败

**可能原因**：
- PDF 文件损坏
- 包含扫描图像（非文本）
- 编码问题

**解决方案**：
- 确保 PDF 是文本格式
- 尝试重新下载 PDF
- 检查 PDF 是否能在其他阅读器中正常打开

### 问题 4：向量数据库错误

**解决方案**：

```bash
# 清理并重建向量数据库
python cleanup.py
# 选择 "清理所有数据"
# 然后重新上传文档
```

### 快速验证

运行验证脚本检查系统状态：

```bash
python quick_test.py
```

如果所有测试通过，说明系统正常。

## 📊 性能优化

### 当前性能指标

- PDF 解析速度：约 5-10 秒/15 页
- 向量化速度：约 1-2 秒/1000 字符
- 检索延迟：< 500ms
- 问答延迟：取决于 LLM API（通常 2-5 秒）

### 优化建议

详见 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)，包括：

1. **Rerank 重排序**：提升检索准确率
2. **混合检索**：结合关键词和语义检索
3. **层级化切片**：保留文档结构信息
4. **元数据过滤**：按作者、年份等筛选
5. **多模态支持**：图表和公式识别

## 🎯 使用场景

### 学术研究

- 快速了解论文核心内容
- 对比不同论文的方法
- 查找特定技术细节
- 生成文献综述素材

### 学习提升

- 理解复杂的学术概念
- 学习领域知识
- 论文阅读辅助
- 知识点快速查询

### 项目应用

- 技术调研
- 方案设计参考
- 实现细节查询
- 性能指标对比

## 📈 项目亮点

### 技术能力展示

- ✅ RAG 系统设计与实现
- ✅ LangChain 框架应用
- ✅ 向量数据库使用（Chroma）
- ✅ NLP 文本处理（Embedding、分块）
- ✅ 前端开发（Streamlit）
- ✅ 系统集成与优化
- ✅ 问题诊断与修复能力

### 适合简历展示

**项目描述示例**：

> 基于 RAG 技术的科研论文智能问答系统
>
> - 使用 LangChain + ChromaDB 构建检索增强生成（RAG）系统
> - 实现 PDF 文档自动解析、向量化和持久化存储
> - 集成 Sentence Transformers 进行语义相似度检索
> - 开发 Streamlit Web 界面，提供友好的交互体验
> - 实现引文溯源功能，答案可追溯到原文档段落
> - 诊断并修复库版本兼容性问题，提升系统稳定性
>
> 技术栈：Python, LangChain, ChromaDB, Streamlit, Sentence Transformers, OpenAI API

## 🗂️ 相关文档

- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md) - 数据管理详细说明
- [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) - 系统优化计划
- [BUGFIX_README.md](BUGFIX_README.md) - 问题修复记录

## 🔄 更新日志

### v1.1.0 (2024-12-30)

- ✅ 修复 sentence-transformers 版本兼容性问题
- ✅ 增强文档验证和 Unicode 字符处理
- ✅ 固定所有依赖包版本，提升稳定性
- ✅ 添加快速验证脚本和依赖修复脚本
- ✅ 完善文档和故障排除指南

### v1.0.0 (2024-12-30)

- 🎉 初始版本发布
- ✅ 基础 RAG 功能实现
- ✅ PDF 文档处理
- ✅ 向量检索与问答
- ✅ 引文溯源功能
- ✅ Streamlit Web 界面
- ✅ 数据管理功能

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👨‍💻 作者

[@Your Name](https://github.com/yourusername)

## 🙏 致谢

- [LangChain](https://www.langchain.com/) - 强大的 LLM 应用框架
- [ChromaDB](https://www.trychroma.com/) - 优秀的向量数据库
- [Sentence Transformers](https://www.sbert.net/) - 高质量的文本 Embedding 模型
- [Streamlit](https://streamlit.io/) - 简洁易用的 Web 框架

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [项目 Issues 页面](https://github.com/yourusername/rag-paper/issues)
- Email: your.email@example.com

---

⭐ 如果这个项目对你有帮助，请给个 Star！
