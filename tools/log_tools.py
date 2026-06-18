# -*- coding: utf-8 -*-
"""
tools/log_tools.py
==================
日志分析工具集：把非结构化的控制台输出，转化为**结构化指标与错误诊断**，
为大模型的决策提供量化依据（区别于让模型"肉眼"看几千行日志）。

两个工具
--------
1. parse_train_log    : 提取 loss / accuracy / 学习率 的数值序列，
                        判断收敛趋势（下降/上升/震荡/发散）。
2. parse_error_stack  : 解析 Python/PyTorch 报错堆栈，归类为
                        维度不匹配 / 显存不足(OOM) / 语法错误 / 梯度异常 /
                        设备不匹配 / 类型错误 等，并定位错误行号。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.config import CONFIG
from core.tool_registry import tool


# ============================================================
# 1. 训练日志指标解析
# ============================================================
# 兼容多种常见打印格式：loss=0.23 / loss: 0.23 / "loss" 0.23 / train_loss 0.23
_FLOAT = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
_LOSS_PAT = re.compile(r"(?:train[_ ]?|val[_ ]?|valid[_ ]?)?loss\s*[=:：]?\s*" + _FLOAT, re.I)
_ACC_PAT = re.compile(r"(?:acc(?:uracy)?|正确率|准确率)\s*[=:：]?\s*" + _FLOAT, re.I)
_LR_PAT = re.compile(r"(?:lr|learning[_ ]?rate|学习率)\s*[=:：]?\s*" + _FLOAT, re.I)


def _extract_series(text: str, pattern: re.Pattern) -> List[float]:
    vals = []
    for m in pattern.finditer(text):
        try:
            vals.append(float(m.group(1)))
        except (ValueError, IndexError):
            continue
    return vals


def _trend(series: List[float]) -> str:
    """根据数值序列判断变化趋势。"""
    if len(series) < 2:
        return "数据点不足，无法判断趋势"
    first, last = series[0], series[-1]
    # 检测 NaN / Inf 发散
    if any(v != v for v in series) or any(abs(v) == float("inf") for v in series):
        return "发散（出现 NaN/Inf，训练异常）"
    # 末值相对首值变化
    if first == 0:
        change = last - first
    else:
        change = (last - first) / abs(first)
    # 震荡判断：统计相邻点的方向反转次数（拐点比例），而非单纯幅度
    directions = [
        1 if series[i + 1] > series[i] else (-1 if series[i + 1] < series[i] else 0)
        for i in range(len(series) - 1)
    ]
    nonzero = [d for d in directions if d != 0]
    reversals = sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1])
    volatile = len(nonzero) >= 3 and reversals >= max(2, len(nonzero) * 0.4)
    if change < -0.1:
        base = "持续下降（收敛中）"
    elif change > 0.1:
        base = "持续上升"
    else:
        base = "基本平稳"
    return base + ("，但存在明显震荡" if volatile else "")


@tool(description="解析训练日志文本，结构化提取 loss/accuracy/学习率 的数值序列并判断收敛趋势")
def parse_train_log(log_text: str = "", from_last_run: bool = False) -> str:
    """解析训练日志，输出结构化指标与趋势判断。

    Args:
        log_text: 训练日志原始文本；若为空且 from_last_run=True 则读取上次运行的日志
        from_last_run: 是否直接读取上次 run_train_script 保存的输出日志
    """
    if from_last_run or not log_text.strip():
        log_path = CONFIG.paths.logs / "last_train_output.log"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        elif not log_text.strip():
            return "[提示] 未提供日志文本，且未找到上次运行日志（logs/last_train_output.log）"

    losses = _extract_series(log_text, _LOSS_PAT)
    accs = _extract_series(log_text, _ACC_PAT)
    lrs = _extract_series(log_text, _LR_PAT)

    lines = ["# 训练日志结构化分析"]

    if losses:
        lines.append(
            f"- Loss: 共 {len(losses)} 个点，首值 {losses[0]:.4g} → 末值 {losses[-1]:.4g}，"
            f"最小 {min(losses):.4g}，趋势：{_trend(losses)}"
        )
    else:
        lines.append("- Loss: 未在日志中检测到 loss 数值")

    if accs:
        lines.append(
            f"- Accuracy: 共 {len(accs)} 个点，首值 {accs[0]:.4g} → 末值 {accs[-1]:.4g}，"
            f"最高 {max(accs):.4g}，趋势：{_trend(accs)}"
        )
    else:
        lines.append("- Accuracy: 未检测到准确率数值")

    if lrs:
        lines.append(f"- 学习率: 范围 {min(lrs):.2g} ~ {max(lrs):.2g}（末值 {lrs[-1]:.2g}）")

    # 综合判断
    verdict = []
    if losses:
        if "发散" in _trend(losses):
            verdict.append("⚠ 训练发散，疑似学习率过大或数据/标签异常")
        elif "下降" in _trend(losses):
            verdict.append("✓ Loss 正常下降，训练有效")
        elif "平稳" in _trend(losses) and losses[0] == losses[-1]:
            verdict.append("⚠ Loss 几乎不变，疑似梯度未回传或学习率过小")
    if verdict:
        lines.append("# 诊断结论: " + "；".join(verdict))
    return "\n".join(lines)


# ============================================================
# 2. 报错堆栈解析与分类
# ============================================================
# 错误类型规则库：(分类名, 匹配关键字列表, 建议)
_ERROR_RULES: List[Tuple[str, List[str], str]] = [
    ("显存不足 (CUDA OOM)",
     ["out of memory", "CUDA out of memory", "OutOfMemoryError"],
     "减小 batch_size、降低模型规模，或切换 CPU / 启用梯度累积"),
    ("张量维度不匹配",
     ["size mismatch", "shapes cannot be multiplied", "must match the size",
      "Expected.*dimension", "mat1 and mat2", "RuntimeError: The size of tensor",
      "dimension out of range"],
     "检查层的输入/输出维度、view/reshape 参数、Linear 的 in_features"),
    ("设备不匹配 (CPU/GPU)",
     ["Expected all tensors to be on the same device",
      "Input type.*and weight type", "device_type", "cuda:0 and cpu"],
     "确保模型与数据用 .to(device) 放到同一设备"),
    ("梯度异常",
     ["nan", "inf", "does not require grad", "element 0 of tensors",
      "gradient", "retain_graph", "backward through the graph a second time"],
     "检查学习率、是否漏写 optimizer.zero_grad()/loss.backward()、是否出现除零/log(0)"),
    ("数据类型错误",
     ["expected scalar type", "Long but found Float", "dtype", "Float but found Long",
      "expected.*Long"],
     "用 .long()/.float() 转换张量类型，分类标签通常需 LongTensor"),
    ("语法错误",
     ["SyntaxError", "IndentationError", "invalid syntax", "unexpected indent"],
     "检查缩进、括号、冒号等 Python 语法"),
    ("模块/导入错误",
     ["ModuleNotFoundError", "ImportError", "No module named"],
     "pip 安装缺失的依赖包，或检查 import 路径"),
    ("属性/名称错误",
     ["AttributeError", "NameError", "has no attribute", "is not defined"],
     "检查变量/方法名拼写、对象是否正确初始化"),
    ("索引/键错误",
     ["IndexError", "KeyError", "list index out of range", "out of bounds"],
     "检查索引范围、字典键、数据集长度"),
    ("文件/路径错误",
     ["FileNotFoundError", "No such file", "cannot find"],
     "检查数据集/权重文件路径是否存在"),
]


@tool(description="解析 Python/PyTorch 报错堆栈，自动归类错误类型（维度/显存/梯度/类型/语法等）、定位行号并给出修复方向")
def parse_error_stack(error_text: str = "", from_last_run: bool = False) -> str:
    """解析报错信息并分类。

    Args:
        error_text: 报错堆栈文本；为空且 from_last_run=True 时读取上次运行日志
        from_last_run: 是否读取上次 run_train_script 保存的输出日志
    """
    if from_last_run or not error_text.strip():
        log_path = CONFIG.paths.logs / "last_train_output.log"
        if log_path.exists():
            error_text = log_path.read_text(encoding="utf-8", errors="replace")
        elif not error_text.strip():
            return "[提示] 未提供报错文本，且未找到上次运行日志"

    if not error_text.strip():
        return "未提供报错内容"

    low = error_text.lower()

    # 命中分类
    matched: List[Tuple[str, str]] = []
    for cat, keywords, advice in _ERROR_RULES:
        for kw in keywords:
            if re.search(kw.lower(), low):
                matched.append((cat, advice))
                break

    # 提取异常类型行（最后一个 Error/Exception 行通常是根因）
    exc_line = ""
    for line in reversed(error_text.strip().splitlines()):
        if re.search(r"(Error|Exception)\b", line):
            exc_line = line.strip()
            break

    # 提取出错文件与行号
    locations = re.findall(r'File "([^"]+)", line (\d+)(?:, in (\S+))?', error_text)

    lines = ["# 报错堆栈分析"]
    if exc_line:
        lines.append(f"- 异常摘要: {exc_line}")
    if matched:
        # 去重保持顺序
        seen = set()
        lines.append("- 错误归类:")
        for cat, advice in matched:
            if cat in seen:
                continue
            seen.add(cat)
            lines.append(f"    · {cat} → 修复方向: {advice}")
    else:
        lines.append("- 错误归类: 未匹配到已知类型，建议结合异常摘要与知识库进一步分析")
    if locations:
        lines.append("- 出错位置（调用栈，越靠后越接近根因）:")
        for f, ln, fn in locations[-5:]:
            fname = Path(f).name
            where = f" 函数 {fn}" if fn else ""
            lines.append(f"    · {fname}:{ln}{where}")
    return "\n".join(lines)


if __name__ == "__main__":
    from core.tool_registry import registry
    print("log_tools 已注册:",
          [t for t in registry.list_tools() if t in ("parse_train_log", "parse_error_stack")])
    print("\n--- 训练日志解析 ---")
    demo_log = """
    Epoch 1 step 10 loss=2.31 acc=0.12 lr=0.001
    Epoch 1 step 20 loss=1.85 acc=0.34 lr=0.001
    Epoch 1 step 30 loss=1.20 acc=0.56 lr=0.001
    Epoch 1 step 40 loss=0.74 acc=0.78 lr=0.001
    """
    print(registry.execute("parse_train_log", {"log_text": demo_log}))
    print("\n--- 报错堆栈解析 ---")
    demo_err = '''Traceback (most recent call last):
  File "train.py", line 42, in <module>
    out = model(x)
  File "model.py", line 18, in forward
    x = self.fc(x)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x784 and 256x10)'''
    print(registry.execute("parse_error_stack", {"error_text": demo_err}))
