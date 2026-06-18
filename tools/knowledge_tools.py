# -*- coding: utf-8 -*-
"""
tools/knowledge_tools.py
========================
把 RAG 检索能力封装成 ReAct 可调用的工具，并提供全局唯一的 RAGEngine 单例
（避免每次检索都重新加载向量模型）。
"""

from typing import Optional

from core.tool_registry import tool
from knowledge.rag_engine import RAGEngine

# 全局单例：首次使用时初始化
_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


@tool(description="检索 PyTorch 调试领域知识库（课件/官方规范/故障排查手册），返回与问题最相关的若干条专业知识，用于增强调试建议的准确性")
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """从本地知识库检索与问题相关的专业知识。

    Args:
        query: 检索问题或关键词（如 "loss 不收敛"、"CUDA out of memory"）
        top_k: 返回的相关知识条数，默认 3
    """
    engine = get_engine()
    if engine.count() == 0:
        return ("[提示] 知识库为空，请先运行 `python -m knowledge.init_kb` 初始化知识库。"
                "当前可基于通用知识回答。")
    hits = engine.query(query, top_k=top_k)
    if not hits:
        return f"知识库中未检索到与 '{query}' 相关的内容。"
    blocks = [f"检索到 {len(hits)} 条相关知识：\n"]
    for i, h in enumerate(hits, start=1):
        blocks.append(f"【知识 {i}】{h.format()}")
    return "\n\n".join(blocks)


@tool(description="查看知识库当前状态（后端类型、已入库文本块数量）")
def knowledge_base_stats() -> str:
    """返回知识库统计信息。无参数。"""
    return get_engine().stats()
