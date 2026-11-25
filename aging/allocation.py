"""合同分配核对与复查情景下的可行安排。

约束（来自业务约定）：

- 同一家族用于同一合同的数量受 ``family_cap`` 限制；``mix_allowed=0`` 时只能
  用单一批，且数量必须满足 ``need``。
- 已确认（confirmed=1）的持有关系是既有约束；已发运（shipped=1）的长单保留为
  先前承诺，不存在于 picks 的"确认"一律不得当作约束。
- 复查抽走的件从可发库存中扣除；同一实物不能在两个合同的假设方案里重复占用。
- 复查回执晚于合同 depart_day 时，对该合同不可用；未观测结果只展开为分支，
  不能拿最有利分支同时保证两个竞争合同。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fusion import FusedEvidence, fuse_result, fused_segment_risk
from .recheck import LabTest, RecheckPlan, RecheckResult, check_coverage
from .risk import (
    BORDERLINE,
    NO_EVIDENCE,
    PASS,
    REJECT,
    REFERENCE_ONLY,
    FitRegistry,
    Lot,
    SegmentRisk,
    assess_segment,
)

# 只有 pass 是"合格批"；临界、不合格、证据不足、薄池化、仅参考证据都不得放行。
ELIGIBLE_VERDICTS = frozenset({PASS})


@dataclass
class LotState:
    lot: Lot
    gross_qty: int
    committed: int                       # 已确认持有占用（含已发运）
    consumed_recheck: int = 0
    segment_risks: dict[str, SegmentRisk] = field(default_factory=dict)
    fused: dict[str, FusedEvidence] = field(default_factory=dict)

    @property
    def available(self) -> int:
        return max(0, self.gross_qty - self.committed - self.consumed_recheck)


@dataclass
class AllocationLine:
    lot_id: str
    family: str
    qty: int
    verdict: str
    point: float
    ucb95: float
    estimability: str


@dataclass
class ContractFeasibility:
    contract_id: str
    need: int
    mix_allowed: bool
    family_cap: int
    feasible: bool
    reason: str
    lines: list[AllocationLine]
    shortfall: int


@dataclass
class PlanEvaluation:
    plan: RecheckPlan
    asof_day: int
    states: dict[str, LotState]
    contracts: dict[str, ContractFeasibility]
    # 未观测复查的分支可行性：test_id -> {"pass_branch": ..., "fail_branch": ...}
    branches: dict[str, dict[str, ContractFeasibility]]
    coverage_warnings: list[str]
    # 竞争合同在同一假设下的库存占用核对
    double_use_conflicts: list[str]


# ---------------------------------------------------------------------------
# 库存状态
# ---------------------------------------------------------------------------


def build_states(bundle, lots: list[Lot], plan: RecheckPlan) -> dict[str, LotState]:
    committed: dict[str, int] = {}
    for row in bundle.table("picks"):
        if row["confirmed"] != "1":
            continue  # 不存在的确认不得当作约束
        committed[row["lot"]] = committed.get(row["lot"], 0) + int(row["qty"])
    states: dict[str, LotState] = {}
    for lot in lots:
        states[lot.id] = LotState(
            lot=lot,
            gross_qty=lot.qty,
            committed=committed.get(lot.id, 0),
            consumed_recheck=plan.consumed(lot.id),
        )
    return states


def grade_condition_match(lot: Lot, contract: dict[str, str]) -> bool:
    return lot.grade == contract["grade"] and lot.condition == contract["condition"]


def annotate_risks(states: dict[str, LotState], contracts: list[dict[str, str]],
                   registry: FitRegistry) -> None:
    for state in states.values():
        for contract in contracts:
            if not grade_condition_match(state.lot, contract):
                continue
            fit_bundle = registry.for_lot(state.lot)
            state.segment_risks[contract["id"]] = assess_segment(
                fit_bundle, state.lot, contract,
                direct_bundle=registry.direct(state.lot),
            )


# ---------------------------------------------------------------------------
# 单合同可行性（回溯枚举，数据规模小，优先可解释而非求解器黑盒）
# ---------------------------------------------------------------------------


def _eligible_lots(states: dict[str, LotState], contract: dict[str, str],
                   verdict_override: dict[str, str] | None = None) -> list[LotState]:
    """返回该合同下判定为 pass 的库存状态；override 用于复查分支假设。"""
    out = []
    for state in states.values():
        lot = state.lot
        if not grade_condition_match(lot, contract):
            continue
        risk = state.segment_risks.get(contract["id"])
        if risk is None:
            continue
        verdict = (verdict_override or {}).get(lot.id, risk.verdict)
        if verdict in ELIGIBLE_VERDICTS and state.available > 0:
            out.append(state)
    return out


def evaluate_contract(states: dict[str, LotState], contract: dict[str, str],
                      verdict_override: dict[str, str] | None = None) -> ContractFeasibility:
    need = int(contract["need"])
    family_cap = int(contract["family_cap"])
    mix_allowed = contract["mix_allowed"] == "1"
    eligible = _eligible_lots(states, contract, verdict_override)

    lines: list[AllocationLine] = []
    if not mix_allowed:
        # 不可混装：必须单批满足 need，且同批数量受 family_cap 约束。
        candidates = [
            state for state in eligible
            if min(state.available, family_cap) >= need
        ]
        if not candidates:
            return ContractFeasibility(
                contract["id"], need, mix_allowed, family_cap, False,
                "没有单一合格批能满足 need（不可混装）或受同家族上限约束",
                [], need,
            )
        state = max(candidates, key=lambda item: item.available)
        risk = state.segment_risks[contract["id"]]
        lines.append(AllocationLine(
            state.lot.id, state.lot.family, need, risk.verdict,
            risk.point, risk.ucb95, risk.estimability))
        return ContractFeasibility(
            contract["id"], need, mix_allowed, family_cap, True,
            "单批满足", lines, 0)

    # 可混装：按段风险点估计从低到高贪心，再校验家族上限与总量。
    remaining = need
    family_used: dict[str, int] = {}
    ordered = sorted(eligible, key=lambda state: state.segment_risks[contract["id"]].point)
    for state in ordered:
        if remaining == 0:
            break
        family = state.lot.family
        headroom = family_cap - family_used.get(family, 0)
        take = min(state.available, headroom, remaining)
        if take <= 0:
            continue
        risk = state.segment_risks[contract["id"]]
        lines.append(AllocationLine(
            state.lot.id, family, take, risk.verdict,
            risk.point, risk.ucb95, risk.estimability))
        family_used[family] = family_used.get(family, 0) + take
        remaining -= take
    feasible = remaining == 0
    reason = "混装满足" if feasible else f"合格库存不足，缺口 {remaining} 件"
    return ContractFeasibility(
        contract["id"], need, mix_allowed, family_cap, feasible, reason, lines, remaining)


# ---------------------------------------------------------------------------
# 整案评估：多合同同时求解 + 分支 + 重复占用核对
# ---------------------------------------------------------------------------


def evaluate_plan(bundle, plan: RecheckPlan, contracts: list[dict[str, str]],
                  lots: list[Lot], registry: FitRegistry, asof_day: int,
                  tests: dict[str, LabTest],
                  results: dict[str, RecheckResult],
                  rng_seed: int = 20261125) -> PlanEvaluation:
    states = build_states(bundle, lots, plan)
    annotate_risks(states, contracts, registry)

    known = available_result_ids(plan, asof_day, results)
    fusion_notes = _apply_fusion(
        plan, tests, results, known, states, contracts, registry, rng_seed
    )
    feasibility = {
        contract["id"]: evaluate_contract(states, contract) for contract in contracts
    }
    coverage_warnings = _coverage_warnings(plan, tests, states, contracts, known)
    coverage_warnings.extend(fusion_notes)
    branches = _branch_feasibility(plan, tests, results, asof_day, states, contracts)
    conflicts = _double_use_check(plan, tests, results, asof_day, states, contracts)
    return PlanEvaluation(
        plan=plan,
        asof_day=asof_day,
        states=states,
        contracts=feasibility,
        branches=branches,
        coverage_warnings=coverage_warnings,
        double_use_conflicts=conflicts,
    )


def _apply_fusion(plan: RecheckPlan, tests: dict[str, LabTest],
                  results: dict[str, RecheckResult], known: set[str],
                  states: dict[str, LotState], contracts: list[dict[str, str]],
                  registry: FitRegistry, rng_seed: int) -> list[str]:
    """把已到回执融合进对应批的段风险；只覆盖 tested_age 之内的窗口。"""
    import dataclasses

    import numpy as np

    rng = np.random.default_rng(rng_seed + plan_hash(plan))
    notes: list[str] = []
    for test_id in plan.test_ids:
        if test_id not in known:
            continue
        test = tests[test_id]
        result = results[test_id]
        state = states[test.lot]
        fit_bundle = registry.for_lot(state.lot)
        fused = fuse_result(fit_bundle, test, result)
        state.fused[test_id] = fused
        for contract in contracts:
            risk = state.segment_risks.get(contract["id"])
            if risk is None:
                continue
            if risk.end_age > fused.covers_window_end:
                notes.append(
                    f"{test_id} 回执（{result.failed}/{result.checked} 失效，CP95 上界 "
                    f"{result.cpb_upper95:.3f}）只覆盖到 {fused.covers_window_end} 天，"
                    f"{contract['id']} 窗口末端 {risk.end_age} 天超出，维持生存模型判定")
                continue
            point, ucb = fused_segment_risk(
                fused, risk.start_age, risk.end_age,
                fit_bundle.bootstrap_rates, rng,
            )
            new_verdict = _verdict_with_level(
                risk.evidence_level, point, ucb,
                risk.risk_limit, risk.estimability)
            state.segment_risks[contract["id"]] = dataclasses.replace(
                risk, point=point, ucb95=ucb,
                verdict_override=new_verdict, fused_from=test_id,
            )
            notes.append(
                f"{test_id} 回执已融合进 {test.lot} 对 {contract['id']} 的段风险："
                f"点估计 {point:.4f} / UCB95 {ucb:.4f}，判定 {new_verdict}")
    return notes


def plan_hash(plan: RecheckPlan) -> int:
    return sum(ord(char) for char in ",".join(plan.test_ids))


def _verdict_with_level(level: str, point: float, ucb: float,
                        limit: float, estimability: str) -> str:
    from .risk import (
        FULL_POOLED,
        THIN_CURRENT_POOL,
        UNESTIMABLE,
    )
    if estimability == UNESTIMABLE:
        return NO_EVIDENCE
    if level in (FULL_POOLED, "reference_pooled"):
        return REFERENCE_ONLY
    if level == THIN_CURRENT_POOL:
        # 复查证据直接来自该批，可把薄池化升级为数值判定，但仍要过限值。
        pass
    if point > limit:
        return REJECT
    if ucb > limit:
        return BORDERLINE
    return PASS


def available_result_ids(plan: RecheckPlan, asof_day: int,
                         results: dict[str, RecheckResult]) -> set[str]:
    return {
        test_id for test_id in plan.test_ids
        if test_id in results and results[test_id].received_day <= asof_day
    }


def _coverage_warnings(plan: RecheckPlan, tests: dict[str, LabTest],
                       states: dict[str, LotState], contracts: list[dict[str, str]],
                       known: set[str]) -> list[str]:
    warnings: list[str] = []
    for test_id in plan.test_ids:
        test = tests[test_id]
        state = states[test.lot]
        for contract in contracts:
            risk = state.segment_risks.get(contract["id"])
            if risk is None:
                continue
            full_cover, gap = check_coverage(test, risk.end_age)
            if not full_cover:
                status = "已收回执" if test_id in known else "结果未观测"
                warnings.append(
                    f"{test_id}（{status}）只检查到 {test.tested_age} 天龄期，"
                    f"{contract['id']} 窗口末端 {risk.end_age} 天尚有 {gap} 天"
                    "不被该快速检查覆盖，不能据此声明整个保障期风险")
    return warnings


def _branch_feasibility(plan: RecheckPlan, tests: dict[str, LabTest],
                        results: dict[str, RecheckResult], asof_day: int,
                        states: dict[str, LotState],
                        contracts: list[dict[str, str]]) -> dict[str, dict[str, ContractFeasibility]]:
    branches: dict[str, dict[str, ContractFeasibility]] = {}
    for test_id in plan.test_ids:
        test = tests[test_id]
        if test_id in results and results[test_id].received_day <= asof_day:
            continue  # 已有事实，不是分支
        # 分支只影响该复查对应批的判定，两个分支都必须独立可执行。
        pass_override = {test.lot: PASS}
        fail_override = {test.lot: REJECT}
        branches[test_id] = {
            "pass_branch": {
                contract["id"]: evaluate_contract(states, contract, pass_override)
                for contract in contracts
            },
            "fail_branch": {
                contract["id"]: evaluate_contract(states, contract, fail_override)
                for contract in contracts
            },
        }
    return branches


def _double_use_check(plan: RecheckPlan, tests: dict[str, LabTest],
                      results: dict[str, RecheckResult], asof_day: int,
                      states: dict[str, LotState],
                      contracts: list[dict[str, str]]) -> list[str]:
    """同一实物不得在两个竞争合同的同一假设分支里同时被占用。"""
    conflicts: list[str] = []
    # 基准（已知证据）下的跨合同占用核对。
    usage: dict[str, list[str]] = {}
    for contract in contracts:
        feasibility = evaluate_contract(states, contract)
        if not feasibility.feasible:
            continue
        for line in feasibility.lines:
            usage.setdefault(line.lot_id, []).append(
                f"{contract['id']}:{line.qty}")
    for lot_id, claims in usage.items():
        if len(claims) > 1:
            state = states[lot_id]
            claimed = sum(int(item.split(":")[1]) for item in claims)
            if claimed > state.available:
                conflicts.append(
                    f"批 {lot_id} 可发 {state.available} 件，却被 {', '.join(claims)} "
                    "同时占用，必须按合同优先级串行分配，不能重复保证")
    # 未观测复查的通过分支不能同时解救两个合同。
    for test_id in plan.test_ids:
        if test_id in results and results[test_id].received_day <= asof_day:
            continue
        test = tests[test_id]
        pass_override = {test.lot: PASS}
        pass_usage: dict[str, int] = {}
        for contract in contracts:
            feasibility = evaluate_contract(states, contract, pass_override)
            if not feasibility.feasible:
                continue
            for line in feasibility.lines:
                if line.lot_id == test.lot:
                    pass_usage[contract["id"]] = line.qty
        if len(pass_usage) > 1:
            total = sum(pass_usage.values())
            state = states[test.lot]
            if total > state.available:
                conflicts.append(
                    f"未观测复查 {test_id} 的通过分支被 {len(pass_usage)} 个合同同时"
                    f"用于保证批 {test.lot}（共需 {total} 件 > 可发 {state.available} 件）")
    return conflicts


# ---------------------------------------------------------------------------
# 决策截止与迟到回执
# ---------------------------------------------------------------------------


def latest_substitution_deadline(contract: dict[str, str],
                                 tests: dict[str, LabTest]) -> dict[str, int | None]:
    """每个复查动作若要影响该合同，最晚何时必须作出替代决定。

    回执 read_at 晚于 depart_day 的动作对该合同不可用，截止点取
    min(depart_day, read_at)；完全赶不上的动作显式返回 None。
    """
    depart_day = int(contract["depart_day"])
    deadlines: dict[str, int | None] = {}
    for test_id, test in tests.items():
        if test.read_at > depart_day:
            deadlines[test_id] = None
        else:
            deadlines[test_id] = min(depart_day, test.read_at)
    return deadlines


def shipped_lot_notifications(evaluation: PlanEvaluation, bundle,
                              contracts: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """已出库批若段风险不达标，提示应联系的客户；只改现行解释，不撤销出库事实。

    回执 received 晚于合同 depart_day 时，标记为发运后到达：它更新现行解释，
    但不改变旧时点的可行性与已发生的出库。
    """
    depart_by_contract = {
        row["id"]: int(row["depart_day"]) for row in (contracts or [])
    }
    shipped = [row for row in bundle.table("picks") if row["shipped"] == "1"]
    notifications = []
    for row in shipped:
        state = evaluation.states.get(row["lot"])
        if state is None:
            continue
        risk = state.segment_risks.get(row["contract"])
        if risk is None or risk.verdict == PASS:
            continue
        timing = ""
        if risk.fused_from:
            received = evaluation.plan.results[risk.fused_from].received_day
            depart = depart_by_contract.get(row["contract"])
            if depart is not None and received > depart:
                timing = (f"复查 {risk.fused_from} 于 day {received} 到达，晚于该合同 "
                          f"day {depart} 的发运；只更新解释，不改旧时点预测")
        notifications.append({
            "contract": row["contract"],
            "lot": row["lot"],
            "qty": row["qty"],
            "verdict": risk.verdict,
            "point": f"{risk.point:.4f}",
            "ucb95": f"{risk.ucb95:.4f}",
            "limit": f"{risk.risk_limit:.2f}",
            "fused_from": risk.fused_from or "",
            "late_timing": timing,
            "action": "联系客户提示保障期风险；出库事实不撤销，仅更新现行解释",
        })
    return notifications
