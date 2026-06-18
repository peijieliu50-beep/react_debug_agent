# -*- coding: utf-8 -*-
"""
web/deploy_guard.py
===================
公开部署安全模块：为 Streamlit 应用提供
    1) 访问密码门（未输入正确口令看不到任何内容、也调不动 API）
    2) 全局每日调用次数上限（即便口令泄露，也限制 API 烧钱额度）

密钥与口令来源（优先级）：
    Streamlit Secrets（云端，st.secrets） > 环境变量 / .env（本地）

设计说明
--------
- 限额采用**基于文件的全局计数器**（按 UTC 日期存一个 JSON），
  云端多会话共享同一文件，跨用户累计，真正起到“总闸”作用。
- 本模块只在公开部署时启用；本地运行若未设置口令，默认放行（不影响开发）。
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


# ------------------------------------------------------------
# 读取密钥 / 口令 / 限额（Secrets 优先，回退环境变量）
# ------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    # st.secrets 在未配置 secrets 文件时访问会抛异常，需保护
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # noqa: BLE001
        pass
    return os.getenv(key, default)


def inject_api_key() -> bool:
    """把 Secrets/环境变量里的 DeepSeek 密钥注入 os.environ，供 config 读取。
    返回是否成功拿到密钥。"""
    key = _get_secret("DEEPSEEK_API_KEY", "")
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
        # 同步刷新已加载的 CONFIG 单例
        try:
            from config.config import CONFIG
            CONFIG.api_key = key
        except Exception:  # noqa: BLE001
            pass
        return True
    return False


def get_access_password() -> str:
    """访问口令；为空表示未设置（本地开发默认放行）。"""
    return _get_secret("ACCESS_PASSWORD", "")


def get_daily_limit() -> int:
    """每日全局调用上限；<=0 表示不限制。"""
    try:
        return int(_get_secret("DAILY_CALL_LIMIT", "0") or "0")
    except ValueError:
        return 0


# ------------------------------------------------------------
# 密码门
# ------------------------------------------------------------
def check_password() -> bool:
    """渲染密码门。返回 True 表示已通过（或无需口令）。

    未设置 ACCESS_PASSWORD 时直接放行（方便本地开发）。
    """
    pwd = get_access_password()
    if not pwd:
        return True  # 未设口令 → 本地开发，放行

    if st.session_state.get("_authed"):
        return True

    st.markdown("## 🔒 访问验证")
    st.caption("本应用为课程作品演示，请输入访问口令。")
    entered = st.text_input("访问口令", type="password", key="_pwd_input")
    col1, _ = st.columns([1, 4])
    with col1:
        ok = st.button("进入")
    if ok:
        if entered == pwd:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("口令错误，请重试。")
    st.stop()  # 阻断后续界面渲染
    return False


# ------------------------------------------------------------
# 全局每日调用限额（基于文件计数）
# ------------------------------------------------------------
def _counter_file() -> Path:
    base = Path(os.getenv("TMPDIR", "/tmp")) if os.name != "nt" else Path(os.getenv("TEMP", "."))
    base.mkdir(parents=True, exist_ok=True)
    return base / "react_agent_call_counter.json"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def remaining_quota() -> int:
    """返回今日剩余可用次数；不限额时返回一个大数。"""
    limit = get_daily_limit()
    if limit <= 0:
        return 10 ** 9
    fp = _counter_file()
    used = 0
    if fp.exists():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if data.get("date") == _today():
                used = int(data.get("count", 0))
        except Exception:  # noqa: BLE001
            used = 0
    return max(0, limit - used)


def consume_quota() -> bool:
    """消耗一次配额。返回 True=允许调用，False=已超限。"""
    limit = get_daily_limit()
    if limit <= 0:
        return True
    fp = _counter_file()
    data = {"date": _today(), "count": 0}
    if fp.exists():
        try:
            old = json.loads(fp.read_text(encoding="utf-8"))
            if old.get("date") == _today():
                data["count"] = int(old.get("count", 0))
        except Exception:  # noqa: BLE001
            pass
    if data["count"] >= limit:
        return False
    data["count"] += 1
    try:
        fp.write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return True
