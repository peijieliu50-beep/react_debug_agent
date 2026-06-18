# Web 界面启动与使用说明

## 一、启动方式

确保已安装依赖（见根目录 `requirements.txt`）并配置好 API 密钥后，在**项目根目录**执行：

```bash
streamlit run web/app.py
```

浏览器会自动打开（或手动访问终端提示的 `http://localhost:8501`）。

> 若 8501 端口被占用，可指定端口：`streamlit run web/app.py --server.port 8600`

## 二、首次使用准备

1. **配置 API 密钥**（二选一）
   - 在左侧配置区"DeepSeek API 密钥"输入框直接填写；
   - 或在项目根目录 `.env` 中设置 `DEEPSEEK_API_KEY`（参考 `.env.example`）。
2. **初始化知识库**（可选但推荐）
   ```bash
   python -m knowledge.init_kb --reset
   ```
   也可在界面左侧"上传资料入库"直接拖入 PDF/DOCX/MD/TXT。

## 三、界面分区（三栏布局）

| 区域 | 功能 |
|------|------|
| **左栏 · 配置区** | API 密钥、采样温度、最大迭代轮数、Debug 模式开关、知识库上传与状态 |
| **中栏 · 主对话区** | 聊天式交互；可多文件上传（代码/报错截图）；实时展示四层信息 |
| **右栏 · 辅助面板** | 总迭代轮数、推理进度条、训练 Loss 趋势简图、当前状态 |

## 四、四层信息分色（体现 ReAct 可解释性）

| 颜色 | 含义 |
|------|------|
| 🔵 蓝色 | 用户输入 |
| 🟣 紫色 | Agent 思考（Thought） |
| 🟢 绿色 | 工具执行（Action + Observation） |
| 🔴 红色 | 最终结论（Final Answer） |

每一轮 Thought→Action→Observation 都会实时渲染成卡片，答辩时可直观演示智能体内部决策逻辑。

## 五、调试报告导出

一次会话结束后，点击底部"📥 导出调试报告 (Markdown)"，
即可下载包含**完整推理轨迹、代码修改点、训练验证记录、最终结论**的 Markdown 报告。

## 六、常见问题

- **提示未配置密钥**：在左栏填写 API 密钥即可。
- **Windows OpenMP 报错**：程序已内置 `KMP_DUPLICATE_LIB_OK=TRUE` 兜底，一般无需处理。
- **知识库显示 fallback 后端**：表示未安装 chromadb/sentence-transformers，已自动降级为本地关键词检索，功能正常；安装后重启即自动启用向量检索。
- **OCR 不可用**：需额外安装 Tesseract 引擎并在 `.env` 配置 `TESSERACT_CMD`。
