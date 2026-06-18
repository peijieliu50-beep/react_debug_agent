# -*- coding: utf-8 -*-
"""
config/config.py
================
全局配置中心：统一管理大模型 API、ReAct 调度参数、知识库、训练执行、
文档解析及路径等全部可调参数。

【安全设计】
    API 密钥**不硬编码**，统一通过环境变量 `DEEPSEEK_API_KEY` 读取；
    本地开发可在项目根目录放置 `.env` 文件（已加入 .gitignore），由
    python-dotenv 自动加载，避免密钥泄露。

用法：
    from config.config import CONFIG
    api_key = CONFIG.api_key
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------
# 0. 自动加载 .env（若存在）
# ------------------------------------------------------------
try:
    from dotenv import load_dotenv

    # 项目根目录 = 本文件的上一级目录
    _ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_ROOT / ".env")
except ImportError:
    # 未安装 python-dotenv 时不报错，仅依赖系统环境变量
    _ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# 1. 路径配置（全部基于项目根目录，跨平台通用）
# ------------------------------------------------------------
@dataclass
class Paths:
    root: Path = _ROOT
    core: Path = _ROOT / "core"
    tools: Path = _ROOT / "tools"
    knowledge: Path = _ROOT / "knowledge"
    assets: Path = _ROOT / "assets"
    web: Path = _ROOT / "web"
    docs: Path = _ROOT / "docs"
    # 向量库持久化目录
    vector_store: Path = _ROOT / "knowledge" / "vector_store"
    # 原始知识文档目录（PDF/DOCX/TXT）
    knowledge_docs: Path = _ROOT / "knowledge" / "docs"
    # Agent 运行时的工作区（被调试代码、训练产物落地于此）
    workspace: Path = _ROOT / "assets" / "workspace"
    # 运行日志目录
    logs: Path = _ROOT / "logs"


# ------------------------------------------------------------
# 2. 全局配置
# ------------------------------------------------------------
@dataclass
class Config:
    # ---------- 大模型 API（DeepSeek，兼容 OpenAI 接口） ----------
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )
    # 采样温度：调试场景需稳定、可复现，取较低值
    temperature: float = 0.2
    # 单次回复最大 token
    max_tokens: int = 4096

    # ---------- ReAct 调度参数 ----------
    # Thought→Action→Observation 闭环最大迭代步数，防止死循环
    max_iterations: int = 15
    # 单次大模型调用超时（秒）
    request_timeout: int = 120
    # 调用失败 / 工具执行报错的自动重试次数
    max_retries: int = 3

    # ---------- 工具执行参数 ----------
    # 训练 / 代码执行子进程超时（秒），默认 10 分钟，防止训练卡死
    exec_timeout: int = 600
    # debug 模式：CPU 演示环境下自动裁剪训练规模
    debug_mode: bool = True
    debug_max_steps: int = 3000      # 训练总步数上限，3-5 分钟可跑完一轮
    debug_batch_size: int = 8        # 缩小 batch size
    debug_max_epochs: int = 3        # 训练轮数（小数据下足够观察收敛）
    # 是否自动检测 CUDA，无显卡降级 CPU
    auto_device: bool = True

    # ---------- 知识库 / 向量检索 ----------
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
        )
    )
    collection_name: str = "pytorch_debug_kb"
    # 检索返回的相关片段数量
    retrieve_top_k: int = 3
    # 文档切分块大小（字符）与重叠
    chunk_size: int = 512
    chunk_overlap: int = 128

    # ---------- 文档解析 ----------
    # Tesseract OCR 引擎可执行文件路径（Windows 需显式指定，留空则用 PATH）
    tesseract_cmd: str = field(
        default_factory=lambda: os.getenv("TESSERACT_CMD", "")
    )
    ocr_lang: str = "chi_sim+eng"     # OCR 识别语言：简体中文+英文

    # ---------- 路径 ----------
    paths: Paths = field(default_factory=Paths)

    # ---------- 校验与初始化 ----------
    def ensure_dirs(self) -> None:
        """确保运行所需目录存在（向量库、工作区、日志、知识文档目录）。"""
        for p in (
            self.paths.vector_store,
            self.paths.knowledge_docs,
            self.paths.workspace,
            self.paths.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def validate_api(self) -> bool:
        """检查 API 密钥是否已通过环境变量配置。"""
        return bool(self.api_key)


# ------------------------------------------------------------
# 3. 全局单例
# ------------------------------------------------------------
CONFIG = Config()
CONFIG.ensure_dirs()


if __name__ == "__main__":
    # 快速自检：python -m config.config
    print("项目根目录 :", CONFIG.paths.root)
    print("模型名称   :", CONFIG.model_name)
    print("API 地址   :", CONFIG.base_url)
    print("最大迭代步 :", CONFIG.max_iterations)
    print("API 密钥已配置:", "是" if CONFIG.validate_api() else "否（请设置环境变量 DEEPSEEK_API_KEY）")
