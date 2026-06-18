# -*- coding: utf-8 -*-
"""
tools/file_tools.py
===================
文件操作工具集：Agent 与代码环境交互的基础"手脚"。

包含 read_file / write_file / list_dir / search_in_file 四个工具，
全部经过**路径白名单校验**，仅允许操作项目内的受信目录，杜绝越级访问系统文件。

【安全设计】
    _resolve_safe() 把任意输入路径解析为绝对路径后，校验其必须位于白名单根目录
    （工作区 workspace、素材 assets、知识库 knowledge）之内，否则拒绝执行。
    相对路径默认相对于"工作区"目录。
"""

from pathlib import Path
from typing import List

from config.config import CONFIG
from core.tool_registry import tool


# ------------------------------------------------------------
# 路径白名单
# ------------------------------------------------------------
# 允许操作的根目录（解析后的绝对路径）
_ALLOWED_ROOTS: List[Path] = [
    CONFIG.paths.workspace.resolve(),
    CONFIG.paths.assets.resolve(),
    CONFIG.paths.knowledge.resolve(),
]
# 相对路径的默认基准目录
_DEFAULT_BASE: Path = CONFIG.paths.workspace.resolve()


class PathNotAllowedError(Exception):
    """越权访问白名单之外路径时抛出。"""


def _resolve_safe(path: str) -> Path:
    """把输入路径解析为绝对路径并做白名单校验。"""
    p = Path(path)
    if not p.is_absolute():
        p = _DEFAULT_BASE / p
    p = p.resolve()
    for root in _ALLOWED_ROOTS:
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    allowed = " | ".join(str(r) for r in _ALLOWED_ROOTS)
    raise PathNotAllowedError(
        f"路径越权：'{path}' 不在白名单目录内。仅允许操作：{allowed}"
    )


# ------------------------------------------------------------
# 工具：读取文件
# ------------------------------------------------------------
@tool(description="读取项目内指定文本文件的内容，可限制最大行数并附带行号")
def read_file(path: str, max_lines: int = 300, with_line_no: bool = True) -> str:
    """读取文件内容。

    Args:
        path: 文件路径（相对工作区或白名单内的绝对路径）
        max_lines: 最多读取的行数，防止超长文件刷屏
        with_line_no: 是否输出行号（便于定位 bug 行）
    """
    fp = _resolve_safe(path)
    if not fp.exists():
        return f"[错误] 文件不存在: {path}"
    if not fp.is_file():
        return f"[错误] 不是文件: {path}"

    lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    shown = lines[:max_lines]
    if with_line_no:
        body = "\n".join(f"{i + 1:>4} | {ln}" for i, ln in enumerate(shown))
    else:
        body = "\n".join(shown)
    suffix = "" if total <= max_lines else f"\n...（共 {total} 行，已显示前 {max_lines} 行）"
    return f"# 文件: {fp.name}（{total} 行）\n{body}{suffix}"


# ------------------------------------------------------------
# 工具：写入 / 修改文件
# ------------------------------------------------------------
@tool(description="写入或覆盖项目内文件内容（用于修复后保存代码），自动备份原文件")
def write_file(path: str, content: str, backup: bool = True) -> str:
    """写入文件内容（覆盖式）。

    Args:
        path: 目标文件路径（相对工作区或白名单内绝对路径）
        content: 要写入的完整文本内容
        backup: 写入前是否备份原文件为 .bak
    """
    fp = _resolve_safe(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    backup_note = ""
    if backup and fp.exists():
        bak = fp.with_suffix(fp.suffix + ".bak")
        bak.write_text(fp.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        backup_note = f"（原文件已备份至 {bak.name}）"
    fp.write_text(content, encoding="utf-8")
    n = len(content.splitlines())
    return f"[成功] 已写入 {fp.name}，共 {n} 行 {backup_note}"


# ------------------------------------------------------------
# 工具：列目录
# ------------------------------------------------------------
@tool(description="列出项目内指定目录下的文件与子目录")
def list_dir(path: str = ".", show_hidden: bool = False) -> str:
    """列出目录内容。

    Args:
        path: 目录路径（相对工作区或白名单内绝对路径），默认工作区根
        show_hidden: 是否显示以 . 开头的隐藏文件
    """
    dp = _resolve_safe(path)
    if not dp.exists():
        return f"[错误] 目录不存在: {path}"
    if not dp.is_dir():
        return f"[错误] 不是目录: {path}"

    entries = []
    for item in sorted(dp.iterdir(), key=lambda x: (x.is_file(), x.name)):
        if not show_hidden and item.name.startswith("."):
            continue
        if item.is_dir():
            entries.append(f"[DIR ] {item.name}/")
        else:
            size = item.stat().st_size
            entries.append(f"[FILE] {item.name}  ({size} bytes)")
    if not entries:
        return f"目录 {dp.name}/ 为空"
    return f"# 目录: {dp}\n" + "\n".join(entries)


# ------------------------------------------------------------
# 工具：文件内搜索
# ------------------------------------------------------------
@tool(description="在指定文件中按关键字/正则搜索，返回匹配的行号与内容（用于定位代码）")
def search_in_file(path: str, keyword: str, use_regex: bool = False, max_hits: int = 30) -> str:
    """在文件中搜索关键字。

    Args:
        path: 文件路径
        keyword: 要搜索的关键字或正则表达式
        use_regex: 是否按正则匹配
        max_hits: 最多返回的匹配行数
    """
    import re

    fp = _resolve_safe(path)
    if not fp.exists() or not fp.is_file():
        return f"[错误] 文件不存在: {path}"

    lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = []
    pattern = re.compile(keyword) if use_regex else None
    for i, ln in enumerate(lines, start=1):
        matched = pattern.search(ln) if use_regex else (keyword in ln)
        if matched:
            hits.append(f"{i:>4} | {ln}")
            if len(hits) >= max_hits:
                break
    if not hits:
        return f"未在 {fp.name} 中找到 '{keyword}'"
    return f"# 在 {fp.name} 中匹配 '{keyword}'（{len(hits)} 处）:\n" + "\n".join(hits)


if __name__ == "__main__":
    # 自检
    from core.tool_registry import registry
    print("file_tools 已注册:", [t for t in registry.list_tools()
                                  if t in ("read_file", "write_file", "list_dir", "search_in_file")])
    print(registry.execute("write_file", {"path": "demo.txt", "content": "hello\nworld"}))
    print(registry.execute("read_file", {"path": "demo.txt"}))
    print(registry.execute("list_dir", {"path": "."}))
    print(registry.execute("search_in_file", {"path": "demo.txt", "keyword": "world"}))
    # 越权测试
    print(registry.execute("read_file", {"path": "C:/Windows/system.ini"}))
