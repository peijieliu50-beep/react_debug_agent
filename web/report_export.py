# -*- coding: utf-8 -*-
"""
web/report_export.py
====================
把一次 ReAct 调试会话的完整轨迹导出为 Markdown 调试报告，
包含：用户请求、逐轮思考/动作/观察、识别出的代码修改点、最终验证结论。
"""

from datetime import datetime
from typing import List

from core.react_engine import ReActResult, ReActStep


def _collect_modifications(steps: List[ReActStep]) -> List[str]:
    """从轨迹中提取代码修改点（write_file 动作）。"""
    mods = []
    for s in steps:
        if s.action == "write_file":
            path = s.action_input.get("path", "未知文件")
            mods.append(f"- `{path}`：{s.observation.strip()[:80]}")
    return mods


def _collect_runs(steps: List[ReActStep]) -> List[str]:
    """提取训练运行验证记录（run_train_script 动作）。"""
    runs = []
    for s in steps:
        if s.action == "run_train_script":
            script = s.action_input.get("script_path", "脚本")
            status = "成功" if "成功 (exit=0)" in s.observation else "失败/异常"
            runs.append(f"- 运行 `{script}` → {status}")
    return runs


def build_markdown_report(
    result: ReActResult,
    user_query: str,
    title: str = "PyTorch 自动化调试报告",
    timestamp: str = "",
) -> str:
    """生成 Markdown 格式的调试报告。

    Args:
        result: ReAct 执行结果
        user_query: 用户原始请求
        title: 报告标题
        timestamp: 时间戳字符串（由调用方传入，避免在引擎内取系统时间）
    """
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mods = _collect_modifications(result.steps)
    runs = _collect_runs(result.steps)

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("> 课程：人工智能导论 课程设计（大作业）  ")
    lines.append("> 选题：基于 ReAct 范式的 PyTorch 深度学习实验自动化调试 Agent  ")
    lines.append(f"> 生成时间：{ts}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 一、调试请求
    lines.append("## 一、调试请求")
    lines.append("")
    lines.append("```")
    lines.append(user_query.strip())
    lines.append("```")
    lines.append("")

    # 二、概要
    lines.append("## 二、调试概要")
    lines.append("")
    lines.append(f"- 是否成功：{'✅ 是' if result.success else '⚠ 未完全完成'}")
    lines.append(f"- 总迭代轮数：{result.iterations}")
    lines.append(f"- 终止原因：{_reason_cn(result.stopped_reason)}")
    lines.append(f"- 代码修改点：{len(mods)} 处")
    lines.append(f"- 训练验证次数：{len(runs)} 次")
    lines.append("")

    # 三、完整推理轨迹
    lines.append("## 三、完整 ReAct 推理轨迹")
    lines.append("")
    for s in result.steps:
        lines.append(f"### 第 {s.index} 轮")
        if s.thought:
            lines.append(f"**💭 Thought（推理）**")
            lines.append("")
            lines.append(f"{s.thought}")
            lines.append("")
        if s.action:
            import json
            args = json.dumps(s.action_input, ensure_ascii=False)
            lines.append(f"**🔧 Action（行动）**：调用工具 `{s.action}`")
            lines.append("")
            lines.append(f"```json\n{args}\n```")
            lines.append("")
        if s.observation:
            obs = s.observation if len(s.observation) < 1500 else s.observation[:1500] + "\n...(已截断)"
            lines.append(f"**👁️ Observation（观察）**")
            lines.append("")
            lines.append(f"```\n{obs}\n```")
            lines.append("")
        if s.is_final:
            lines.append("**✅ 本轮给出最终回答**")
            lines.append("")
        lines.append("")

    # 四、修改点汇总
    lines.append("## 四、代码修改点汇总")
    lines.append("")
    if mods:
        lines.extend(mods)
    else:
        lines.append("（本次会话未修改代码文件）")
    lines.append("")

    # 五、训练验证记录
    lines.append("## 五、训练验证记录")
    lines.append("")
    if runs:
        lines.extend(runs)
    else:
        lines.append("（本次会话未运行训练验证）")
    lines.append("")

    # 六、最终结论
    lines.append("## 六、最终结论")
    lines.append("")
    lines.append(result.final_answer.strip() or "（无最终结论）")
    lines.append("")

    return "\n".join(lines)


def _reason_cn(reason: str) -> str:
    return {
        "final_answer": "模型给出最终答案",
        "max_iterations": "达到最大迭代步数",
        "error": "调度过程出错",
    }.get(reason, reason)
