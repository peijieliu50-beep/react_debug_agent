# -*- coding: utf-8 -*-
"""
core/llm_client.py
==================
DeepSeek 大模型客户端封装。

特性
----
- 基于官方 `openai` SDK，通过 `base_url` 指向 DeepSeek，**兼容 OpenAI 接口格式**。
- 支持**非流式**（chat）与**流式**（chat_stream）两种调用。
- 原生 **Function Calling**：传入 tools schema，自动解析模型返回的工具调用请求。
- 健壮性：调用失败/超时**自动重试**（指数退避），统一异常处理。

【安全】API 密钥仅来自 `config.CONFIG.api_key`（环境变量），不在代码中硬编码。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from config.config import CONFIG

try:
    from openai import OpenAI
    from openai import APIError, APITimeoutError, APIConnectionError, RateLimitError
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "未安装 openai SDK，请先执行: pip install openai>=1.30.0"
    ) from e


# ------------------------------------------------------------
# 工具调用请求的标准化结构
# ------------------------------------------------------------
@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""
    id: str
    name: str
    arguments: str          # 原始 JSON 字符串（参数）


@dataclass
class LLMResponse:
    """一次（非流式）模型响应的标准化封装。"""
    content: str = ""                       # 文本内容（Thought / Final Answer）
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    raw_message: Any = None                 # 原始 message 对象（用于回填对话）

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class DeepSeekClient:
    """DeepSeek Chat 客户端（OpenAI 兼容）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or CONFIG.api_key
        self.base_url = base_url or CONFIG.base_url
        self.model = model or CONFIG.model_name

        if not self.api_key:
            raise RuntimeError(
                "未检测到 API 密钥。请设置环境变量 DEEPSEEK_API_KEY，"
                "或在项目根目录的 .env 文件中配置。"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=CONFIG.request_timeout,
            max_retries=0,        # 重试由本类自行控制，便于记录日志
        )

    # ------------------------------------------------------------
    # 非流式调用（ReAct 主循环使用）
    # ------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """发起一次对话补全，返回标准化响应。

        Args:
            messages: 对话消息列表
            tools: Function Calling 的工具 schema 列表
            temperature: 采样温度，默认取配置
            max_tokens: 最大生成 token，默认取配置
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else CONFIG.temperature,
            "max_tokens": max_tokens or CONFIG.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._with_retry(lambda: self._client.chat.completions.create(**kwargs))
        return self._parse_response(resp)

    # ------------------------------------------------------------
    # 流式调用（前端实时展示文本时使用）
    # ------------------------------------------------------------
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """流式返回文本增量（不含工具调用，用于纯文本输出展示）。"""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else CONFIG.temperature,
            "max_tokens": CONFIG.max_tokens,
            "stream": True,
        }
        stream = self._with_retry(lambda: self._client.chat.completions.create(**kwargs))
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ------------------------------------------------------------
    # 内部：解析响应
    # ------------------------------------------------------------
    @staticmethod
    def _parse_response(resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: List[ToolCall] = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments or "{}",
                    )
                )
        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            raw_message=msg,
        )

    # ------------------------------------------------------------
    # 内部：带重试的调用（指数退避）
    # ------------------------------------------------------------
    def _with_retry(self, call):
        last_err: Optional[Exception] = None
        for attempt in range(CONFIG.max_retries + 1):
            try:
                return call()
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                last_err = e
                wait = min(2 ** attempt, 8)
                if attempt < CONFIG.max_retries:
                    time.sleep(wait)
                continue
            except APIError as e:
                # 4xx 类错误（如参数错误）重试无意义，直接抛出
                last_err = e
                if getattr(e, "status_code", 500) and 400 <= getattr(e, "status_code", 500) < 500:
                    break
                wait = min(2 ** attempt, 8)
                if attempt < CONFIG.max_retries:
                    time.sleep(wait)
                continue
        raise RuntimeError(f"DeepSeek API 调用失败（已重试 {CONFIG.max_retries} 次）: {last_err}")


if __name__ == "__main__":
    # 自检：需已配置 DEEPSEEK_API_KEY
    if not CONFIG.validate_api():
        print("未配置 DEEPSEEK_API_KEY，跳过实际调用，仅验证类可实例化逻辑。")
    else:
        client = DeepSeekClient()
        r = client.chat([{"role": "user", "content": "用一句话介绍你自己"}])
        print("模型回复:", r.content)
