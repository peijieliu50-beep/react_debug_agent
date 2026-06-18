# 基于 ReAct 范式的 PyTorch 深度学习实验自动化调试 Agent

> 华南理工大学《人工智能导论》课程设计（大作业）
> 选题方向：方向二 —— 科研工具与代码开发类 Agent

一个采用 **ReAct（推理 Reasoning + 行动 Acting）闭环范式**的智能体：它能读懂一段出错的
PyTorch 训练代码，自主完成 **Thought（推理）→ Action（调用工具）→ Observation（观察结果）**
的多轮循环，定位并修复 bug，再实际运行训练验证修复效果，最终给出可解释的调试报告。

---

## ✨ 核心特性

- **ReAct 闭环自主调试**：不止于"问答"，能真正读代码、改代码、跑代码、看结果，循环迭代直至修复。
- **可解释推理轨迹**：每一步 Thought / Action / Observation 全程留痕，适合课程答辩演示。
- **DeepSeek 大模型驱动**：基于 `deepseek-chat`，原生 Function Calling，兼容 OpenAI 接口，128K 上下文。
- **本地知识库增强（RAG）**：Chroma 向量库检索 PyTorch 常见报错与最佳实践，提升修复准确率。
- **多格式资料解析**：PyPDF2 / python-docx / pytesseract(OCR)，本地预处理为纯文本喂给模型。
- **纯 CPU 友好**：训练执行工具内置 debug 模式，自动裁剪训练步数与 batch size，自动检测 CUDA，无显卡降级 CPU。
- **密钥安全**：API 密钥仅从环境变量读取，不硬编码。

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                      web/  Streamlit 交互界面                   │
│            （上传代码 / 提问 → 展示推理轨迹与修复结果）           │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│                  core/  ReAct 核心调度层                        │
│   ┌──────────┐   ┌───────────┐   ┌────────────┐                │
│   │ Thought  │──▶│  Action   │──▶│Observation │── 循环 ◀──────┐ │
│   │ (LLM推理)│   │(选择工具) │   │ (工具结果) │               │ │
│   └──────────┘   └─────┬─────┘   └────────────┘               │ │
│                        │            ▲                          │ │
└────────────────────────┼────────────┼──────────────────────────┘
            │            │      DeepSeek API (Function Calling)
            ▼            │
┌────────────────────────▼──────────────────────────────────────┐
│                       tools/  工具函数集                         │
│  读代码 · 写/改代码 · 静态检查 · 运行训练(debug+CPU降级) ·        │
│  读报错日志 · 知识库检索 · 文档解析(PDF/DOCX/OCR)                 │
└───────────────────────────────┬──────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                                ▼
┌──────────────────┐                          ┌────────────────────┐
│ knowledge/        │                          │ assets/             │
│ Chroma 向量知识库 │                          │ 样例bug代码/工作区   │
└──────────────────┘                          └────────────────────┘

           config/  全局配置（API/迭代步数/超时/debug参数）
```

---

## 📁 目录结构

| 目录 | 职责 |
|------|------|
| `core/`      | ReAct 核心调度层（Thought→Action→Observation 主循环、消息编排） |
| `tools/`     | 工具函数集（代码读写、静态检查、训练执行、知识检索、文档解析） |
| `knowledge/` | 知识库原始文档与 Chroma 向量库 |
| `assets/`    | 测试素材、样例 bug 代码、Agent 运行工作区 |
| `web/`       | Streamlit 交互界面 |
| `config/`    | 全局配置中心 `config.py` |
| `docs/`      | 课程作业报告及配套文档 |

---

## 🚀 环境安装

要求 **Python 3.10+**。

```bash
# 1. （可选）创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 2. 安装依赖（建议使用国内镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置 API 密钥（二选一）
#   方式A：复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY
cp .env.example .env
#   方式B：直接设置环境变量
#   Windows (PowerShell):  $env:DEEPSEEK_API_KEY="sk-xxxx"
#   Linux / macOS:         export DEEPSEEK_API_KEY="sk-xxxx"
```

> **OCR 可选**：如需识别截图中的报错信息，需额外安装 [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) 引擎，
> 并在 `.env` 中设置 `TESSERACT_CMD` 指向其可执行文件。

---

## ▶️ 启动方式

**配置自检**（确认密钥与参数就绪）：

```bash
python -m config.config
```

**启动 Web 演示界面**：

```bash
streamlit run web/app.py
```

浏览器访问终端提示的地址（默认 `http://localhost:8501`）即可使用。

---

## 🔑 关键技术选型依据

| 维度 | 选型 | 依据 |
|------|------|------|
| 智能体范式 | ReAct 闭环 | 推理-行动-观察循环实现自主调试，决策可解释、容错率高 |
| 大模型 | DeepSeek Chat | 原生 Function Calling、兼容 OpenAI 接口、128K 上下文、纯文本低成本 |
| 调度实现 | 纯原生 Python | 不依赖重型 Agent 框架，便于讲清底层原理 |
| 向量库 | Chroma | 轻量本地、无需独立服务、适合单机演示 |
| 前端 | Streamlit | 快速搭建可交互网页 |
| 文档解析 | PyPDF2 / python-docx / pytesseract | 本地预处理为纯文本，不依赖多模态模型 |
| 硬件 | CPU 优先 + 自动 CUDA 检测 | debug 模式裁剪训练规模，无显卡可运行 |

---

*本项目为《人工智能导论》课程设计作品，报告见 `docs/` 目录。*
