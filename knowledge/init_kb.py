# -*- coding: utf-8 -*-
"""
knowledge/init_kb.py
====================
知识库初始化入口：扫描指定目录下的全部文档（.md/.txt/.pdf/.docx/...），
用 file_parser 解析为纯文本后，批量导入 RAGEngine 向量库。

用法
----
    # 导入默认知识文档目录 knowledge/docs/
    python -m knowledge.init_kb

    # 导入自定义目录，并先清空旧库
    python -m knowledge.init_kb --dir 路径 --reset
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from config.config import CONFIG
from knowledge.rag_engine import RAGEngine
from tools.file_parser import parse_file, SUPPORTED_EXT


def collect_documents(doc_dir: Path) -> List[Tuple[str, str]]:
    """扫描目录，解析所有支持格式的文件为 (来源名, 正文) 列表。"""
    docs: List[Tuple[str, str]] = []
    if not doc_dir.exists():
        print(f"[警告] 目录不存在: {doc_dir}")
        return docs
    files = [p for p in sorted(doc_dir.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    if not files:
        print(f"[警告] 目录 {doc_dir} 下未找到可解析文档")
        return docs
    for fp in files:
        text = parse_file(str(fp))
        if text.startswith("["):     # 解析降级/失败
            print(f"  跳过 {fp.name}: {text[:60]}")
            continue
        docs.append((fp.name, text))
        print(f"  已解析 {fp.name}（{len(text)} 字）")
    return docs


def init_knowledge_base(doc_dir: Path = None, reset: bool = False) -> str:
    doc_dir = Path(doc_dir or CONFIG.paths.knowledge_docs)
    print(f"=== 初始化知识库 ===\n文档目录: {doc_dir}")
    engine = RAGEngine()
    if reset:
        engine.reset()
        print("已清空旧知识库")
    docs = collect_documents(doc_dir)
    if not docs:
        return "未导入任何文档。"
    result = engine.add_documents(docs)
    print(result)
    print(engine.stats())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="初始化 RAG 知识库")
    parser.add_argument("--dir", type=str, default="", help="文档目录，默认 knowledge/docs")
    parser.add_argument("--reset", action="store_true", help="导入前清空旧库")
    args = parser.parse_args()
    target = Path(args.dir) if args.dir else None
    init_knowledge_base(target, reset=args.reset)
