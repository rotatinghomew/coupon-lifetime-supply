"""实验室复查证据与情景分支。

纪律：

- 复查抽走的件实际占住库存，不能再算可发。
- ``supplemental`` 是已经收到的回执；``received`` 在项目时钟上晚于合同发运日
  的回执，对该合同的调拨决策不可用（更新的是后续可行安排，不是旧时点预测）。
- ``lab_options`` 里还没有回执的动作是"未来选项"，其结果不得当成现有事实；
  只能展开为通过/不通过两个分支，且一个假设结果不能同时给两个竞争合同做保证。
- 快速检查只在 ``tested_age`` 龄期提供证据；合同窗口末端超出该龄期的部分不被
  这次检查覆盖，必须显式标记。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import beta

from .data import DataBundle


@dataclass(frozen=True)
class LabTest:
    id: str
    recheck: str
    lot: str
    consume: int
    read_at: int          # 结果返回时点（项目时钟天）
    tested_age: int       # 检查覆盖到的贴片年龄
    condition: str


@dataclass(frozen=True)
class RecheckResult:
    test_id: str
    received_day: int
    checked: int
    failed: int
    cpb_upper95: float    # 失效率 Clopper–Pearson 95% 单侧上界

    @property
    def observed_failure_rate(self) -> float:
        return self.failed / self.checked


@dataclass
class RecheckPlan:
    """一组被选择执行的复查动作及其在各时点的状态。"""

    test_ids: list[str]
    consumed_by_lot: dict[str, int] = field(default_factory=dict)
    results: dict[str, RecheckResult] = field(default_factory=dict)

    def consumed(self, lot: str) -> int:
        return self.consumed_by_lot.get(lot, 0)


def _cp_upper(failed: int, checked: int) -> float:
    """Clopper–Pearson 单侧 95% 上界 Beta(f+1, n-f) 的 0.95 分位。"""
    if checked == 0:
        return 1.0
    return float(beta.ppf(0.95, failed + 1, checked - failed))


def load_lab_tests(bundle: DataBundle) -> dict[str, LabTest]:
    tests = {}
    for row in bundle.table("lab_options"):
        tests[row["id"]] = LabTest(
            id=row["id"],
            recheck=row["recheck"],
            lot=row["lot"],
            consume=int(row["consume"]),
            read_at=int(row["read_at"]),
            tested_age=int(row["tested_age"]),
            condition=row["condition"],
        )
    return tests


def load_results(bundle: DataBundle) -> dict[str, RecheckResult]:
    results = {}
    for row in bundle.table("supplemental"):
        checked, failed = int(row["checked"]), int(row["failed"])
        results[row["test"]] = RecheckResult(
            test_id=row["test"],
            received_day=int(row["received"]),
            checked=checked,
            failed=failed,
            cpb_upper95=_cp_upper(failed, checked),
        )
    return results


def available_results(results: dict[str, RecheckResult], asof_day: int) -> dict[str, RecheckResult]:
    """截至项目时钟 asof_day（含）真正到达的回执。"""
    return {test_id: result for test_id, result in results.items()
            if result.received_day <= asof_day}


def build_plan(test_ids: list[str], tests: dict[str, LabTest],
               results: dict[str, RecheckResult]) -> RecheckPlan:
    consumed: dict[str, int] = {}
    for test_id in test_ids:
        test = tests[test_id]
        consumed[test.lot] = consumed.get(test.lot, 0) + test.consume
    known = {test_id: results[test_id] for test_id in test_ids if test_id in results}
    return RecheckPlan(test_ids=list(test_ids), consumed_by_lot=consumed, results=known)


def check_coverage(test: LabTest, window_end_age: int) -> tuple[bool, int]:
    """快速检查是否覆盖合同窗口末端；返回 (是否全覆盖, 未覆盖龄期天数)。"""
    gap = max(0, window_end_age - test.tested_age)
    return gap == 0, gap


def result_verdict(result: RecheckResult, risk_limit: float) -> str:
    """实际回执按与合同同一口径判定：CP 上界对限值。"""
    if result.cpb_upper95 <= risk_limit:
        return "pass"
    if result.observed_failure_rate <= risk_limit < result.cpb_upper95:
        return "borderline"
    return "reject"


def branch_verdict(test: LabTest, risk_limit: float) -> dict[str, str]:
    """尚无回执时只能声明分支，不得预设结果。"""
    return {
        "pass_branch": f"若 {test.id} 回执 CP95 上界 ≤ {risk_limit:.2f}，该批方可凭复查升级",
        "fail_branch": f"若 {test.id} 回执 CP95 上界 > {risk_limit:.2f}，该批退出可选集",
    }


def recheck_window_capacity(bundle: DataBundle) -> list[dict[str, object]]:
    """复查窗口容量与可处理批的清单，供排程核对。"""
    windows = []
    for row in bundle.table("rechecks"):
        windows.append({
            "id": row["id"],
            "day": int(row["day"]),
            "cap": int(row["cap"]),
            "lot_options": row["lot_options"].split(",") if row["lot_options"] else [],
        })
    return windows


def validate_windows(bundle: DataBundle, tests: dict[str, LabTest]) -> list[str]:
    """检查复查动作是否落在窗口的 lot_options 内、抽离总量是否超窗口容量。"""
    problems: list[str] = []
    windows = {item["id"]: item for item in recheck_window_capacity(bundle)}
    load_by_window: dict[str, int] = {window_id: 0 for window_id in windows}
    for test in tests.values():
        window = windows.get(test.recheck)
        if window is None:
            problems.append(f"{test.id}: 复查窗口 {test.recheck} 不存在")
            continue
        if test.lot not in window["lot_options"]:
            problems.append(f"{test.id}: 批 {test.lot} 不在窗口 {test.recheck} 的 lot_options 内")
        load_by_window[test.recheck] += test.consume
    for window_id, load in load_by_window.items():
        if load > int(windows[window_id]["cap"]):
            problems.append(
                f"窗口 {window_id} 抽离 {load} 件超过容量 {windows[window_id]['cap']}")
    return problems
