# -*- coding: utf-8 -*-
"""
assets/integration_test.py
==========================
全流程联调测试（工具层端到端）。

目的：在不依赖 LLM API 的前提下，按 ReAct 引擎会执行的动作顺序，
**直接驱动各工具**走完完整的 bug 调试闭环，验证核心流程稳定、可复现：

    复制带bug脚本 → 运行(暴露维度错误) → 解析报错 → 读代码 → 修复维度
    → 重跑(暴露LR发散) → 解析日志 → 修复学习率 → 重跑(验证收敛)

每一步都断言关键结果，任一环节失败即报错退出。
"""

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import CONFIG
from core.tool_registry import registry
import tools.file_tools, tools.train_tools, tools.log_tools, tools.knowledge_tools  # noqa: F401


def section(title):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def assert_true(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(f"联调断言失败: {msg}")


def main():
    results = {}

    # ---------- 准备：把带 bug 脚本复制进工作区 ----------
    section("步骤 0：准备测试素材")
    src = CONFIG.paths.assets / "sample_code" / "train_speaker.py"
    dst = CONFIG.paths.workspace / "train_speaker.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"  已复制 {src.name} → workspace/{dst.name}")

    # ---------- 步骤 1：首次运行，暴露维度错误 ----------
    section("步骤 1：运行训练脚本（预期触发维度不匹配）")
    out1 = registry.execute("run_train_script", {"script_path": "train_speaker.py", "debug": True})
    print(out1[-400:])
    has_dim_err = "shapes cannot be multiplied" in out1 or "exit=1" in out1 or "失败" in out1
    assert_true(has_dim_err, "首次运行应触发维度不匹配报错")
    results["step1_error_exposed"] = has_dim_err

    # ---------- 步骤 2：解析报错堆栈 ----------
    section("步骤 2：解析报错堆栈（预期归类=维度不匹配）")
    diag = registry.execute("parse_error_stack", {"from_last_run": True})
    print(diag)
    assert_true("维度不匹配" in diag, "报错应被正确归类为'张量维度不匹配'")
    results["step2_classified"] = "维度不匹配" in diag

    # ---------- 步骤 3：读取代码定位 fc 层 ----------
    section("步骤 3：读取代码定位 bug 行")
    code = registry.execute("read_file", {"path": "train_speaker.py"})
    assert_true("nn.Linear(16 * n_mels * n_frames" in code, "应能读到错误的 Linear 定义")
    results["step3_located"] = True

    # ---------- 步骤 4：修复维度（fc in_features） ----------
    section("步骤 4：修复维度错误（池化后尺寸 //2）")
    fixed_code = dst.read_text(encoding="utf-8").replace(
        "self.fc = nn.Linear(16 * n_mels * n_frames, num_speakers)",
        "self.fc = nn.Linear(16 * (n_mels // 2) * (n_frames // 2), num_speakers)",
    )
    registry.execute("write_file", {"path": "train_speaker.py", "content": fixed_code})
    print("  已修复 fc 层 in_features: 16*n_mels*n_frames → 16*(n_mels//2)*(n_frames//2)")
    results["step4_dim_fixed"] = True

    # ---------- 步骤 5：重跑，维度通过但 LR 过高致发散 ----------
    section("步骤 5：重新运行（维度已修，预期 loss 不收敛/上升）")
    out2 = registry.execute("run_train_script", {"script_path": "train_speaker.py", "debug": True})
    no_dim_err = "shapes cannot be multiplied" not in out2
    assert_true(no_dim_err, "维度错误应已消除")
    log2 = registry.execute("parse_train_log", {"from_last_run": True})
    print(log2)
    # 直接解析 loss 序列：lr 过高时末值不应明显低于首值
    from tools.log_tools import _extract_series, _LOSS_PAT
    losses2 = _extract_series((CONFIG.paths.logs / "last_train_output.log").read_text(encoding="utf-8"), _LOSS_PAT)
    lr_problem = bool(losses2) and (losses2[-1] >= losses2[0] * 0.7)   # 未明显下降
    assert_true(lr_problem, f"应检测到 loss 未正常下降（lr 过高）: {losses2[0]:.3f}→{losses2[-1]:.3f}")
    results["step5_lr_problem_detected"] = lr_problem

    # ---------- 步骤 6：修复学习率 ----------
    section("步骤 6：修复学习率（1.0 → 1e-3）")
    fixed_code2 = dst.read_text(encoding="utf-8").replace("LR = 1.0", "LR = 1e-3")
    registry.execute("write_file", {"path": "train_speaker.py", "content": fixed_code2})
    print("  已修复学习率: LR = 1.0 → 1e-3")
    results["step6_lr_fixed"] = True

    # ---------- 步骤 7：最终验证，loss 正常下降 ----------
    section("步骤 7：最终验证（预期 loss 正常下降，闭环成功）")
    out3 = registry.execute("run_train_script", {"script_path": "train_speaker.py", "debug": True})
    success_run = "成功 (exit=0)" in out3
    assert_true(success_run, "最终运行应成功退出")
    log3 = registry.execute("parse_train_log", {"from_last_run": True})
    print(log3)
    # 直接解析 loss 序列判断收敛：末值应明显低于首值
    losses3 = _extract_series((CONFIG.paths.logs / "last_train_output.log").read_text(encoding="utf-8"), _LOSS_PAT)
    converged = bool(losses3) and (losses3[-1] < losses3[0] * 0.8)
    assert_true(converged, f"最终 loss 应明显下降（训练有效）: {losses3[0]:.3f}→{losses3[-1]:.3f}")
    results["step7_converged"] = converged

    # ---------- 汇总 ----------
    section("联调结果汇总")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  通过 {passed}/{total} 个检查点")
    print("  >>> 完整 bug 调试闭环联调：" + ("成功 ✅" if passed == total else "存在失败 ❌"))
    return passed == total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
