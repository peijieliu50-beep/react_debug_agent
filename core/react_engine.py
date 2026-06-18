# -*- coding: utf-8 -*-
"""
core/react_engine.py
====================
标准 ReAct 执行引擎：驱动 Thought → Action → Observation 的多轮闭环循环，
是整个智能体的"大脑调度中枢"。

执行流程
--------
    用户请求
       │
       ▼
  ┌─────────────────── 循环（最多 max_iterations 轮）───────────────────┐
  │  1) Thought  : 调用 LLM 推理（携带工具 schema），产出思考文本        │
  │  2) Action   : 若 LLM 发起 Function Calling → 用注册器执行该工具      │
  │  3) Observation: 工具返回结果回填进对话，进入下一轮                   │
  │  终止：LLM 不再调用工具 / 输出 "Final Answer:" / 达到最大步数         │
  └────────────────────────────────────────────────────────────────────┘
       │
       ▼
   最终答案 + 完整推理轨迹（可追溯，供前端展示）

可追溯性
--------
每一步（思考、动作、观察）都记录为 ReActStep 并写入日志文件，前端可完整回放。
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config.config import CONFIG
from core.llm_client import DeepSeekClient, LLMResponse
from core.tool_registry import ToolRegistry, registry as default_registry
from core.prompts import build_messages


# ------------------------------------------------------------
# 单步轨迹记录
# ------------------------------------------------------------
@dataclass
class ReActStep:
    index: int                              # 第几轮
    thought: str = ""                       # 本轮推理文本
    action: str = ""                        # 调用的工具名（无则空）
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""                   # 工具返回结果
    is_final: bool = False                  # 是否为最终回答步
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReActResult:
    success: bool
    final_answer: str
    steps: List[ReActStep]
    iterations: int
    stopped_reason: str                     # final_answer / max_iterations / error

    def trajectory_text(self) -> str:
        """把完整轨迹格式化为可读文本（用于日志/展示）。"""
        lines = []
        for s in self.steps:
            lines.append(f"--------- 第 {s.index} 轮 ---------")
            if s.thought:
                lines.append(f"[Thought] {s.thought}")
            if s.action:
                lines.append(f"[Action] {s.action}({json.dumps(s.action_input, ensure_ascii=False)})")
            if s.observation:
                obs = s.observation if len(s.observation) < 800 else s.observation[:800] + " ...(截断)"
                lines.append(f"[Observation] {obs}")
            if s.is_final:
                lines.append(f"[Final Answer] {s.thought}")
        return "\n".join(lines)


# ------------------------------------------------------------
# ReAct 引擎
# ------------------------------------------------------------
class ReActEngine:
    def __init__(
        self,
        client: Optional[DeepSeekClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: Optional[int] = None,
        on_step: Optional[Callable[[ReActStep], None]] = None,
    ) -> None:
        """
        Args:
            client: LLM 客户端，默认新建 DeepSeekClient
            tool_registry: 工具注册器，默认使用全局 registry
            max_iterations: 最大循环轮数，默认取配置
            on_step: 每完成一步的回调（前端实时展示用）
        """
        self.client = client or DeepSeekClient()
        self.registry = tool_registry or default_registry
        self.max_iterations = max_iterations or CONFIG.max_iterations
        self.on_step = on_step
        self.logger = self._setup_logger()

    # ---------- 日志 ----------
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ReActEngine")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            CONFIG.paths.logs.mkdir(parents=True, exist_ok=True)
            log_file = CONFIG.paths.logs / "react_run.log"
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(fh)
            # 同时输出到控制台（强制 UTF-8，避免 Windows GBK 控制台崩溃）
            import sys
            try:
                sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(sh)
        return logger

    # ---------- 主循环 ----------
    def run(
        self,
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None,
        extra_context: str = "",
    ) -> ReActResult:
        """执行一次完整的 ReAct 调试会话。"""
        messages = build_messages(user_query, history=history, extra_context=extra_context)
        tools_schema = self.registry.get_schemas()
        steps: List[ReActStep] = []

        self.logger.info("=" * 60)
        self.logger.info(f"[新会话] {datetime.now().isoformat(timespec='seconds')}")
        self.logger.info(f"[用户请求] {user_query[:200]}")

        for i in range(1, self.max_iterations + 1):
            step = ReActStep(index=i, timestamp=datetime.now().isoformat(timespec="seconds"))

            # ===== 1) Thought：调用 LLM 推理 =====
            try:
                resp: LLMResponse = self.client.chat(messages, tools=tools_schema)
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"[第{i}轮] LLM 调用失败: {e}")
                step.thought = f"LLM 调用失败: {e}"
                steps.append(step)
                return ReActResult(False, f"调度中断：{e}", steps, i, "error")

            step.thought = resp.content.strip()
            if step.thought:
                self.logger.info(f"[第{i}轮] [Thought] {step.thought[:300]}")

            # 把模型本轮的 assistant 消息回填（含可能的 tool_calls）
            messages.append(self._assistant_message(resp))

            # ===== 终止判断 A：没有工具调用 =====
            if not resp.has_tool_calls:
                # 视为最终回答（无论是否带 Final Answer 前缀）
                step.is_final = True
                final = self._extract_final_answer(resp.content)
                steps.append(step)
                self._emit(step)
                self.logger.info(f"[结束] 模型给出最终答案（第{i}轮）")
                return ReActResult(True, final, steps, i, "final_answer")

            # ===== 2) Action：执行工具调用 =====
            # 设计上每轮聚焦一个动作；若模型一次返回多个，逐个执行并回填
            for tc in resp.tool_calls:
                try:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                step.action = tc.name
                step.action_input = args
                self.logger.info(f"[第{i}轮] [Action] {tc.name}({json.dumps(args, ensure_ascii=False)})")

                # ===== 3) Observation：执行并回填 =====
                observation = self._execute_with_retry(tc.name, args)
                step.observation = observation
                self.logger.info(f"[第{i}轮] [Observation] {observation[:300]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

            steps.append(step)
            self._emit(step)

        # ===== 终止判断 B：达到最大步数 =====
        self.logger.warning(f"[结束] 达到最大迭代步数 {self.max_iterations}，强制停止")
        # 让模型基于已有信息给一个收尾总结
        summary = self._force_summary(messages)
        return ReActResult(False, summary, steps, self.max_iterations, "max_iterations")

    # ---------- 辅助 ----------
    def _execute_with_retry(self, name: str, args: Dict[str, Any]) -> str:
        """执行工具，失败时重试若干次。"""
        last = ""
        for attempt in range(CONFIG.max_retries):
            result = self.registry.execute(name, args)
            last = result
            # 注册器内部异常以 [工具执行异常] 开头，可触发重试
            if not result.startswith("[工具执行异常]"):
                return result
            time.sleep(0.5)
        return last

    def _emit(self, step: ReActStep) -> None:
        if self.on_step:
            try:
                self.on_step(step)
            except Exception:  # noqa: BLE001  回调异常不影响主循环
                pass

    @staticmethod
    def _assistant_message(resp: LLMResponse) -> Dict[str, Any]:
        """把 LLM 响应转换为可回填进 messages 的 assistant 消息。"""
        msg: Dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
        if resp.has_tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in resp.tool_calls
            ]
        return msg

    @staticmethod
    def _extract_final_answer(content: str) -> str:
        if not content:
            return "（模型未返回内容）"
        marker = "Final Answer:"
        idx = content.find(marker)
        if idx >= 0:
            return content[idx + len(marker):].strip()
        return content.strip()

    def _force_summary(self, messages: List[Dict[str, Any]]) -> str:
        """达到上限时，要求模型基于现有信息给出收尾总结。"""
        try:
            messages.append({
                "role": "user",
                "content": "已达到最大调试轮数。请基于以上全部信息，直接给出当前结论与建议，以 Final Answer: 开头。",
            })
            resp = self.client.chat(messages)  # 不再给工具，强制收尾
            return self._extract_final_answer(resp.content)
        except Exception as e:  # noqa: BLE001
            return f"已达到最大迭代步数，且收尾总结失败：{e}"


if __name__ == "__main__":
    # 自检（无 API key 时仅验证可导入与轨迹结构）
    from core.tool_registry import tool

    @tool(description="返回两数之和")
    def _add(a: int, b: int) -> int:
        """求和。

        Args:
            a: 加数1
            b: 加数2
        """
        return a + b

    if not CONFIG.validate_api():
        print("未配置 DEEPSEEK_API_KEY，跳过实跑。已注册工具：", default_registry.list_tools())
    else:
        engine = ReActEngine()
        result = engine.run("请计算 12 + 30 等于多少，使用工具验证。")
        print("\n最终答案:", result.final_answer)
        print("\n轨迹:\n", result.trajectory_text())
