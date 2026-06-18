# -*- coding: utf-8 -*-
"""
tools/train_tools.py
====================
训练执行工具集：本 Agent 区别于普通"代码问答机器人"的核心能力——
**实际运行训练脚本并捕获结果**，用于验证代码修改是否真正有效。

核心设计
--------
1. **子进程隔离执行**：用 subprocess 在独立进程中运行目标 Python 脚本，
   崩溃/卡死不影响主程序；完整捕获 stdout + stderr。
2. **debug 模式**：通过环境变量向训练脚本注入裁剪参数
   （DEBUG_MODE / MAX_STEPS / BATCH_SIZE / MAX_EPOCHS / DEVICE），
   样例脚本读取这些变量后自动把总步数裁到 3000、batch_size=8、单轮 epoch，
   3-5 分钟即可跑完，满足演示效率要求。
3. **硬件自适应**：自动检测 CUDA 可用性，无显卡则注入 DEVICE=cpu，强制 CPU 运行。
4. **超时控制**：默认 10 分钟超时（config.exec_timeout），超时强制终止子进程，
   防止训练卡死长期占用资源。

【安全】仅允许执行白名单目录（工作区/素材）内的 .py 脚本。
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from config.config import CONFIG
from core.tool_registry import tool


# ------------------------------------------------------------
# 路径白名单（与 file_tools 一致，仅允许工作区/素材内脚本）
# ------------------------------------------------------------
_ALLOWED_ROOTS = [
    CONFIG.paths.workspace.resolve(),
    CONFIG.paths.assets.resolve(),
]
_DEFAULT_BASE = CONFIG.paths.workspace.resolve()


def _resolve_script(path: str) -> Path:
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
    raise PermissionError(f"脚本路径越权：'{path}' 不在白名单目录内")


# ------------------------------------------------------------
# 硬件检测
# ------------------------------------------------------------
def _detect_device() -> str:
    """检测可用计算设备，返回 'cuda' 或 'cpu'。"""
    if not CONFIG.auto_device:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@tool(description="检测当前运行环境的计算设备（CUDA显卡是否可用），返回设备类型与详情")
def detect_device() -> str:
    """检测计算设备。无参数。"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"检测到 CUDA 可用，GPU: {name}；torch 版本 {torch.__version__}"
        return f"未检测到可用 CUDA，将使用 CPU 运行；torch 版本 {torch.__version__}"
    except ImportError:
        return "未安装 PyTorch，无法检测设备；默认 CPU 模式"


# ------------------------------------------------------------
# 工具：运行训练脚本
# ------------------------------------------------------------
@tool(description="在子进程中实际运行指定的 PyTorch 训练/Python 脚本，捕获完整输出，支持 debug 裁剪与超时终止")
def run_train_script(
    script_path: str,
    debug: bool = True,
    timeout: int = 0,
    extra_args: str = "",
) -> str:
    """运行训练脚本并返回执行结果与完整控制台输出。

    Args:
        script_path: 要运行的 .py 脚本路径（相对工作区或白名单内绝对路径）
        debug: 是否启用 debug 模式（裁剪步数/batch、CPU 兼容），演示建议开启
        timeout: 超时秒数，<=0 则使用配置默认值（600秒）
        extra_args: 传递给脚本的额外命令行参数（空格分隔）
    """
    try:
        script = _resolve_script(script_path)
    except PermissionError as e:
        return f"[安全拦截] {e}"
    if not script.exists():
        return f"[错误] 脚本不存在: {script_path}"
    if script.suffix != ".py":
        return f"[错误] 仅支持运行 .py 脚本，当前: {script.suffix}"

    # 云端低内存环境保护：若标记了 LOW_MEMORY，则跳过真实训练，避免 OOM 拖垮整个应用
    if os.getenv("LOW_MEMORY", "0") == "1":
        return (
            f"# 执行脚本: {script.name}\n"
            "[环境受限] 当前为在线演示（低内存）环境，已跳过实际训练运行以保证服务稳定。\n"
            "在本地运行可获得真实训练结果。请基于代码静态分析、报错知识库检索"
            "给出诊断与修复建议，并说明本地验证方法。"
        )

    timeout = timeout if timeout and timeout > 0 else CONFIG.exec_timeout
    device = _detect_device()

    # 构造子进程环境变量：向训练脚本注入裁剪参数与设备
    env = os.environ.copy()
    env["DEVICE"] = device
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # 规避 Windows 上 PyTorch/NumPy 重复加载 OpenMP 运行时导致的崩溃
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if debug:
        env["DEBUG_MODE"] = "1"
        env["MAX_STEPS"] = str(CONFIG.debug_max_steps)     # 3000
        env["BATCH_SIZE"] = str(CONFIG.debug_batch_size)   # 8
        env["MAX_EPOCHS"] = str(CONFIG.debug_max_epochs)   # 1
    else:
        env["DEBUG_MODE"] = "0"

    cmd = [sys.executable, str(script)]
    if extra_args.strip():
        cmd.extend(extra_args.split())

    header = (
        f"# 执行脚本: {script.name}\n"
        f"# 设备: {device}  | debug模式: {'开启' if debug else '关闭'}"
        f"{' (MAX_STEPS=%s, BATCH_SIZE=%s, EPOCHS=%s)' % (CONFIG.debug_max_steps, CONFIG.debug_batch_size, CONFIG.debug_max_epochs) if debug else ''}\n"
        f"# 超时: {timeout}s\n"
    )

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(script.parent),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") + (e.stderr or "")
        return (
            header
            + f"[超时终止] 脚本运行超过 {timeout}s 被强制终止（疑似卡死或训练规模过大）。\n"
            + "--- 已捕获的部分输出 ---\n"
            + _tail(partial, 3000)
        )
    except Exception as e:  # noqa: BLE001
        return header + f"[执行异常] {type(e).__name__}: {e}"

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    status = "成功 (exit=0)" if proc.returncode == 0 else f"失败 (exit={proc.returncode})"

    # 把本次完整输出落盘，便于日志分析工具读取
    log_path = CONFIG.paths.logs / "last_train_output.log"
    try:
        log_path.write_text(stdout + "\n" + stderr, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    parts = [header, f"# 运行结果: {status}", f"# 完整输出已保存: {log_path.name}"]
    if stdout.strip():
        parts.append("--- STDOUT ---\n" + _tail(stdout, 4000))
    if stderr.strip():
        parts.append("--- STDERR ---\n" + _tail(stderr, 3000))
    if not stdout.strip() and not stderr.strip():
        parts.append("（脚本无任何输出）")
    return "\n".join(parts)


def _tail(text: str, max_chars: int) -> str:
    """保留文本尾部 max_chars 字符（报错通常在末尾）。"""
    if len(text) <= max_chars:
        return text
    return "...(前部已截断)...\n" + text[-max_chars:]


if __name__ == "__main__":
    from core.tool_registry import registry
    print("train_tools 已注册:",
          [t for t in registry.list_tools() if t in ("run_train_script", "detect_device")])
    print(registry.execute("detect_device", {}))
