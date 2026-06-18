# -*- coding: utf-8 -*-
"""
web/app.py
==========
ReAct PyTorch 调试 Agent —— Streamlit 演示界面（课程答辩用）。

界面分层（体现 ReAct 可解释性）：
    ① 用户输入   ② Agent 思考(Thought)   ③ 工具执行(Action/Observation)   ④ 最终结论
四层信息分色展示。

三栏布局：
    左：配置区（API密钥 / 知识库上传 / Debug开关 / 模型参数）
    中：主对话区（聊天 + 多文件上传）
    右：辅助面板（轮次计数 / 训练进度 / loss 趋势简图）

启动：
    streamlit run web/app.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# 确保可导入项目根模块
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 规避 Windows OpenMP 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from config.config import CONFIG
from web.deploy_guard import (
    inject_api_key, check_password, consume_quota, remaining_quota,
    get_daily_limit, get_access_password,
)

# ---- 页面基础配置 ----
st.set_page_config(
    page_title="ReAct PyTorch 调试 Agent",
    page_icon="🛠️",
    layout="wide",
)

# ---- 公开部署安全：注入密钥（Secrets 优先）+ 访问密码门 ----
inject_api_key()          # 从 st.secrets / 环境变量注入 DEEPSEEK_API_KEY
check_password()          # 未设口令则放行；设了口令则拦截未授权访问

# ---- 四层信息配色 ----
COLORS = {
    "user": "#1f77b4",        # 蓝：用户输入
    "thought": "#9467bd",     # 紫：思考过程
    "tool": "#2ca02c",        # 绿：工具执行
    "final": "#d62728",       # 红：最终结论
}


def render_step_card(container, step):
    """把一个 ReActStep 渲染成分色卡片。"""
    if step.thought and not step.is_final:
        container.markdown(
            f"<div style='border-left:4px solid {COLORS['thought']};"
            f"padding:6px 10px;margin:4px 0;background:#f6f0fb;'>"
            f"<b style='color:{COLORS['thought']}'>💭 第{step.index}轮 · 思考</b><br>"
            f"<span style='white-space:pre-wrap'>{_esc(step.thought)}</span></div>",
            unsafe_allow_html=True,
        )
    if step.action:
        import json
        args = json.dumps(step.action_input, ensure_ascii=False)
        obs = step.observation if len(step.observation) < 1200 else step.observation[:1200] + " ...(截断)"
        container.markdown(
            f"<div style='border-left:4px solid {COLORS['tool']};"
            f"padding:6px 10px;margin:4px 0;background:#eefaef;'>"
            f"<b style='color:{COLORS['tool']}'>🔧 第{step.index}轮 · 工具调用</b>: "
            f"<code>{_esc(step.action)}({_esc(args)})</code><br>"
            f"<b>👁️ 观察结果</b>:<br><span style='white-space:pre-wrap;font-size:13px'>{_esc(obs)}</span></div>",
            unsafe_allow_html=True,
        )


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ============================================================
# 会话状态初始化
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []          # [(role, content)]
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "uploaded_context" not in st.session_state:
    st.session_state.uploaded_context = ""


# ============================================================
# 顶部标题
# ============================================================
st.markdown(
    "<h2 style='margin-bottom:0'>🛠️ 基于 ReAct 范式的 PyTorch 深度学习实验自动化调试 Agent</h2>"
    "<p style='color:gray;margin-top:4px'>人工智能导论 · 课程设计（大作业） | 选题：科研工具与代码开发类 Agent</p>",
    unsafe_allow_html=True,
)
st.divider()

# 三栏布局
left, center, right = st.columns([1.1, 2.4, 1.2], gap="medium")


# ============================================================
# 左栏：配置区
# ============================================================
with left:
    st.subheader("⚙️ 配置")

    # 部署模式判定：已设访问口令 = 公开部署，密钥来自服务端 Secrets，隐藏输入框
    _deployed = bool(get_access_password())
    if _deployed:
        st.success("🌐 在线演示模式")
        if get_daily_limit() > 0:
            st.caption(f"今日剩余调用次数：{remaining_quota()} / {get_daily_limit()}")
        if not CONFIG.validate_api():
            st.error("服务端未配置 API 密钥，请联系作者。")
    else:
        api_key = st.text_input(
            "DeepSeek API 密钥",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
            help="留空则读取环境变量 DEEPSEEK_API_KEY",
        )
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            CONFIG.api_key = api_key

    st.markdown("**模型参数**")
    temperature = st.slider("采样温度 temperature", 0.0, 1.0, CONFIG.temperature, 0.1)
    max_iter = st.slider("最大迭代轮数", 3, 25, CONFIG.max_iterations, 1)
    CONFIG.temperature = temperature
    CONFIG.max_iterations = max_iter

    debug_mode = st.toggle("Debug 模式（裁剪训练规模）", value=CONFIG.debug_mode)
    CONFIG.debug_mode = debug_mode

    st.divider()
    st.markdown("**📚 知识库**")
    try:
        from tools.knowledge_tools import get_engine
        kb_info = get_engine().stats()
        st.caption(kb_info)
    except Exception as e:  # noqa: BLE001
        st.caption(f"知识库未就绪: {e}")

    kb_files = st.file_uploader(
        "上传资料入库（PDF/DOCX/MD/TXT）",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        key="kb_upload",
    )
    if kb_files and st.button("➕ 导入知识库"):
        from tools.file_parser import parse_file
        from tools.knowledge_tools import get_engine
        tmp = CONFIG.paths.knowledge_docs
        added = []
        for f in kb_files:
            dst = tmp / f.name
            dst.write_bytes(f.getbuffer())
            text = parse_file(str(dst))
            if not text.startswith("["):
                added.append((f.name, text))
        if added:
            msg = get_engine().add_documents(added)
            st.success(msg)
        else:
            st.warning("未能解析上传的文件")


# ============================================================
# 右栏：辅助面板
# ============================================================
with right:
    st.subheader("📊 运行面板")
    metric_round = st.empty()
    progress_bar = st.empty()
    st.markdown("**Loss 趋势**")
    loss_chart = st.empty()
    st.markdown("**当前状态**")
    status_box = st.empty()

    result = st.session_state.last_result
    if result:
        metric_round.metric("总迭代轮数", result.iterations)
        status_box.info("✅ 已完成" if result.success else "⚠ 未完全完成")
        # 从轨迹中尝试解析最近一次训练日志的 loss 序列画简图
        try:
            from tools.log_tools import _extract_series, _LOSS_PAT
            import pandas as pd
            series = []
            for s in result.steps:
                if s.action == "run_train_script" and s.observation:
                    series = _extract_series(s.observation, _LOSS_PAT)
            if series:
                loss_chart.line_chart(pd.DataFrame({"loss": series}))
            else:
                loss_chart.caption("（暂无训练 loss 数据）")
        except Exception:  # noqa: BLE001
            loss_chart.caption("（暂无训练 loss 数据）")
    else:
        metric_round.metric("总迭代轮数", 0)
        status_box.caption("等待开始调试…")
        loss_chart.caption("（暂无训练 loss 数据）")


# ============================================================
# 中栏：主对话区
# ============================================================
with center:
    st.subheader("💬 调试对话")

    # 历史消息回放
    for role, content in st.session_state.history:
        if role == "user":
            st.markdown(
                f"<div style='border-left:4px solid {COLORS['user']};padding:6px 10px;"
                f"margin:4px 0;background:#eef4fb;'><b style='color:{COLORS['user']}'>🧑 用户</b><br>"
                f"<span style='white-space:pre-wrap'>{_esc(content)}</span></div>",
                unsafe_allow_html=True,
            )
        elif role == "final":
            st.markdown(
                f"<div style='border-left:4px solid {COLORS['final']};padding:6px 10px;"
                f"margin:4px 0;background:#fdeeee;'><b style='color:{COLORS['final']}'>✅ 最终结论</b><br>"
                f"<span style='white-space:pre-wrap'>{_esc(content)}</span></div>",
                unsafe_allow_html=True,
            )

    # 文件上传（随问题一起提供给 Agent 作为上下文）
    up_files = st.file_uploader(
        "上传待调试代码 / 报错截图（可多选）",
        type=["py", "txt", "json", "log", "md", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="chat_upload",
    )

    user_input = st.chat_input("描述你的 PyTorch 代码问题，或粘贴报错信息…")

    if user_input:
        # 解析上传文件为纯文本上下文
        extra_context = ""
        if up_files:
            from tools.file_parser import parse_file
            ws = CONFIG.paths.workspace
            parts = []
            for f in up_files:
                dst = ws / f.name
                dst.write_bytes(f.getbuffer())
                text = parse_file(str(dst))
                parts.append(f"【文件 {f.name}】\n{text[:3000]}")
            extra_context = "\n\n".join(parts)
            st.caption(f"已加载 {len(up_files)} 个文件作为上下文")

        st.session_state.history.append(("user", user_input))
        st.session_state.last_query = user_input

        # 校验密钥
        if not CONFIG.validate_api():
            st.error("未配置 DeepSeek API 密钥，请在左侧填写或设置环境变量 DEEPSEEK_API_KEY。")
        elif not consume_quota():
            st.error(
                f"⚠ 今日演示调用次数已达上限（{get_daily_limit()} 次/天），"
                "为控制成本已暂停。请明天再来，或联系作者。"
            )
        else:
            # 实时展示推理过程
            st.markdown(
                f"<div style='border-left:4px solid {COLORS['user']};padding:6px 10px;"
                f"margin:4px 0;background:#eef4fb;'><b style='color:{COLORS['user']}'>🧑 用户</b><br>"
                f"<span style='white-space:pre-wrap'>{_esc(user_input)}</span></div>",
                unsafe_allow_html=True,
            )
            live_area = st.container()
            prog = right.progress(0, text="Agent 推理中…")

            # 导入全部工具以注册到引擎
            import tools.file_tools, tools.train_tools, tools.log_tools, tools.knowledge_tools  # noqa
            from core.react_engine import ReActEngine

            def _on_step(step):
                render_step_card(live_area, step)
                try:
                    prog.progress(min(step.index / max_iter, 1.0),
                                  text=f"第 {step.index}/{max_iter} 轮…")
                    right_metric = step.index
                except Exception:  # noqa: BLE001
                    pass

            try:
                engine = ReActEngine(max_iterations=max_iter, on_step=_on_step)
                with st.spinner("ReAct 闭环执行中…"):
                    result = engine.run(user_input, extra_context=extra_context)
                prog.progress(1.0, text="完成")

                # 最终结论
                st.markdown(
                    f"<div style='border-left:4px solid {COLORS['final']};padding:6px 10px;"
                    f"margin:4px 0;background:#fdeeee;'><b style='color:{COLORS['final']}'>✅ 最终结论</b><br>"
                    f"<span style='white-space:pre-wrap'>{_esc(result.final_answer)}</span></div>",
                    unsafe_allow_html=True,
                )
                st.session_state.history.append(("final", result.final_answer))
                st.session_state.last_result = result
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"执行出错：{e}")


# ============================================================
# 底部：报告导出
# ============================================================
st.divider()
exp_col1, exp_col2 = st.columns([1, 3])
with exp_col1:
    if st.session_state.last_result:
        from web.report_export import build_markdown_report
        md = build_markdown_report(
            st.session_state.last_result,
            st.session_state.last_query,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        fname = "调试报告_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".md"
        st.download_button("📥 导出调试报告 (Markdown)", md, file_name=fname, mime="text/markdown")
    else:
        st.button("📥 导出调试报告 (Markdown)", disabled=True)
with exp_col2:
    if st.button("🗑️ 清空对话"):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()
