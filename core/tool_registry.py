# -*- coding: utf-8 -*-
"""
core/tool_registry.py
=====================
工具注册器：以**装饰器模式**统一注册所有工具，并依据函数签名与类型注解
**自动生成 OpenAI Function Calling 的 JSON Schema**。

设计目标
--------
1. 新增工具时只需写一个带类型注解和文档字符串的普通函数 + `@tool` 装饰，
   无需改动任何调度逻辑（开闭原则）。
2. 自动从函数签名生成 `parameters` Schema（类型、必填项、说明）。
3. 调用前对参数做**类型校验与自动修正**（如字符串"3"→int 3、"true"→bool），
   格式错误时返回结构化错误信息而非直接崩溃。

用法
----
    from core.tool_registry import registry, tool

    @tool(description="读取项目内指定文件的文本内容")
    def read_file(path: str, max_lines: int = 200) -> str:
        '''读取文件。

        Args:
            path: 相对项目工作区的文件路径
            max_lines: 最多读取的行数
        '''
        ...

    schemas = registry.get_schemas()        # 传给 LLM 的 tools 参数
    result = registry.execute("read_file", {"path": "a.py"})
"""

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, get_args, get_origin, Union


# ------------------------------------------------------------
# Python 类型 → JSON Schema 类型 映射
# ------------------------------------------------------------
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json(annotation: Any) -> Dict[str, Any]:
    """把 Python 类型注解转换为 JSON Schema 片段。"""
    # 无注解：默认为 string
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)

    # Optional[X] / Union[X, None]
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return _python_type_to_json(args[0])
        return {"type": "string"}

    # List[X] / list
    if origin in (list, List):
        item_args = get_args(annotation)
        item_schema = _python_type_to_json(item_args[0]) if item_args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    # dict
    if origin in (dict, Dict):
        return {"type": "object"}

    # 基础类型
    return {"type": _PY_TO_JSON.get(annotation, "string")}


def _parse_arg_docs(docstring: Optional[str]) -> Dict[str, str]:
    """从 Google 风格 docstring 的 Args 段落解析每个参数的说明。"""
    if not docstring:
        return {}
    docs: Dict[str, str] = {}
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if re.match(r"^(Args|Arguments|参数)\s*[:：]?\s*$", stripped):
            in_args = True
            continue
        if in_args:
            # 遇到下一个段落标题则结束
            if re.match(r"^(Returns|Raises|Return|返回|示例|Example)\s*[:：]?", stripped):
                break
            m = re.match(r"^(\w+)\s*[:：]\s*(.+)$", stripped)
            if m:
                docs[m.group(1)] = m.group(2).strip()
    return docs


# ------------------------------------------------------------
# 工具元信息
# ------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any]                 # 完整 function-calling schema
    signature: inspect.Signature
    type_hints: Dict[str, Any]


class ToolRegistry:
    """全局工具注册表。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    # ---------- 注册 ----------
    def register(self, func: Callable = None, *, name: str = None, description: str = None):
        """装饰器：注册一个工具函数。

        可写作 @registry.register 或 @registry.register(description="...")
        """

        def _wrap(f: Callable) -> Callable:
            tool_name = name or f.__name__
            sig = inspect.signature(f)
            doc = inspect.getdoc(f) or ""
            # 工具描述：优先用显式 description，否则取 docstring 首行
            desc = description or (doc.split("\n")[0] if doc else tool_name)
            arg_docs = _parse_arg_docs(doc)

            properties: Dict[str, Any] = {}
            required: List[str] = []
            for pname, param in sig.parameters.items():
                if pname in ("self", "cls"):
                    continue
                prop = _python_type_to_json(param.annotation)
                if pname in arg_docs:
                    prop["description"] = arg_docs[pname]
                properties[pname] = prop
                # 无默认值 → 必填
                if param.default is inspect.Parameter.empty:
                    required.append(pname)

            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=desc,
                func=f,
                schema=schema,
                signature=sig,
                type_hints={
                    p: param.annotation
                    for p, param in sig.parameters.items()
                    if p not in ("self", "cls")
                },
            )
            return f

        # 支持两种调用方式
        if func is not None and callable(func):
            return _wrap(func)
        return _wrap

    # 兼容简洁别名：@tool(...)
    def tool(self, *args, **kwargs):
        return self.register(*args, **kwargs)

    # ---------- 查询 ----------
    def get_schemas(self) -> List[Dict[str, Any]]:
        """返回供 LLM Function Calling 使用的 tools schema 列表。"""
        return [spec.schema for spec in self._tools.values()]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools

    def describe(self) -> str:
        """生成所有工具的文字说明（用于注入系统提示词）。"""
        lines = []
        for spec in self._tools.values():
            params = spec.schema["function"]["parameters"]["properties"]
            pstr = ", ".join(params.keys()) if params else "无"
            lines.append(f"- {spec.name}({pstr}): {spec.description}")
        return "\n".join(lines)

    # ---------- 参数校验与修正 ----------
    def _coerce_value(self, value: Any, annotation: Any) -> Any:
        """尽力把 LLM 传来的参数修正为目标类型。"""
        origin = get_origin(annotation)
        if origin is Union:
            non_none = [a for a in get_args(annotation) if a is not type(None)]
            annotation = non_none[0] if non_none else str

        try:
            if annotation is bool:
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in ("true", "1", "yes", "y", "是")
            if annotation is int:
                return int(float(value)) if not isinstance(value, bool) else value
            if annotation is float:
                return float(value)
            if annotation is str:
                return value if isinstance(value, str) else str(value)
            if annotation in (list, List) or origin in (list, List):
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    # 尝试 JSON 解析，失败则按逗号切分
                    try:
                        parsed = json.loads(value)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except json.JSONDecodeError:
                        return [v.strip() for v in value.split(",") if v.strip()]
                return [value]
            if annotation in (dict, Dict) or origin in (dict, Dict):
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
        except (ValueError, TypeError, json.JSONDecodeError):
            # 修正失败则原样返回，交由后续校验报错
            return value
        return value

    def validate_args(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """校验并修正参数；缺失必填项时抛出 ValueError。"""
        spec = self._tools[name]
        # LLM 偶尔会把参数包成 JSON 字符串
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        coerced: Dict[str, Any] = {}
        for pname, param in spec.signature.parameters.items():
            if pname in ("self", "cls"):
                continue
            if pname in args:
                coerced[pname] = self._coerce_value(args[pname], spec.type_hints.get(pname))
            elif param.default is inspect.Parameter.empty:
                raise ValueError(f"工具 '{name}' 缺少必填参数: '{pname}'")
        return coerced

    # ---------- 执行 ----------
    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """执行工具，返回字符串化结果（始终返回，不向上抛异常）。"""
        if name not in self._tools:
            return f"[错误] 未知工具: '{name}'。可用工具: {', '.join(self.list_tools())}"
        try:
            clean_args = self.validate_args(name, args)
        except ValueError as e:
            return f"[参数错误] {e}"
        try:
            result = self._tools[name].func(**clean_args)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception as e:  # noqa: BLE001  工具内部异常需回传给模型继续推理
            return f"[工具执行异常] {type(e).__name__}: {e}"


# ------------------------------------------------------------
# 全局单例 + 便捷装饰器别名
# ------------------------------------------------------------
registry = ToolRegistry()
tool = registry.register


if __name__ == "__main__":
    # 自检
    @tool(description="两数相加")
    def add(a: int, b: int = 1) -> int:
        """求和。

        Args:
            a: 第一个加数
            b: 第二个加数
        """
        return a + b

    print("已注册工具:", registry.list_tools())
    print("Schema:", json.dumps(registry.get_schemas(), ensure_ascii=False, indent=2))
    print("执行 add(2, '3') =>", registry.execute("add", {"a": 2, "b": "3"}))  # 字符串自动修正
    print("缺参 =>", registry.execute("add", {}))
