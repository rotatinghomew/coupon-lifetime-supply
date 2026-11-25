"""下一轮备货复用：从合同视角回看库存的不可选原因。

输出每张合同下每个匹配批的"可选/不可选"及具体原因，原因互斥且可追溯：

- ``committed``        已确认持有/已发运占用，剩余可发不足
- ``recheck_held``     复查抽离占住，不能再算可发
- ``evidence_thin``    当前生产线证据太薄（薄池化），不能放行
- ``reference_only``   只剩参考/全层证据，不能据此放行
- ``no_evidence``      该层/该龄段无法估计
- ``risk_reject``      点估计超合同风险限值
- ``borderline``       点估计达标但 UCB95 超限
- ``family_cap``       受同家族数量上限约束
- ``eligible``         可进入候选（仍需通过整体可行性求解）
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import DataBundle
from .estimator import build_registry, lots_from
from .recheck import load_lab_tests, load_results
from .risk import (
    BORDERLINE,
    INSUFFICIENT_EVIDENCE,
    NO_EVIDENCE,
    PASS,
    REJECT,
    REFERENCE_ONLY,
)


def _matches(lot, contract: dict[str, str]) -> bool:
    return lot.grade == contract["grade"] and lot.condition == contract["condition"]


@dataclass(frozen=True)
class LotReview:
    contract_id: str
    lot_id: str
    family: str
    available: int
    verdict: str
    reason: str
    point: float
    ucb95: float
    risk_limit: float
    evidence_level: str
    fused_from: str | None


def review_inventory(bundle: DataBundle, asof_day: int,
                     plan_test_ids: list[str] | None = None,
                     n_boot: int = 200, seed: int = 20261125) -> list[LotReview]:
    """从合同视角列出每个匹配批的可选性与不可选原因。"""
    from .allocation import evaluate_plan
    from .data import day_to_calendar
    from .recheck import build_plan

    asof = day_to_calendar(asof_day)
    registry = build_registry(bundle, n_boot=n_boot, seed=seed, asof=asof)
    lots = lots_from(bundle)
    contracts = bundle.table("contracts")
    tests = load_lab_tests(bundle)
    results = load_results(bundle)
    known = {tid: r for tid, r in results.items() if r.received_day <= asof_day}
    plan = build_plan(plan_test_ids or [], tests, known)
    evaluation = evaluate_plan(
        bundle, plan, contracts, lots, registry, asof_day, tests, results
    )

    reviews: list[LotReview] = []
    for contract in contracts:
        cap = int(contract["family_cap"])
        for lot in lots:
            if not _matches(lot, contract):
                continue
            state = evaluation.states[lot.id]
            risk = state.segment_risks.get(contract["id"])
            if risk is None:
                continue
            reason = _reason(state, contract, risk, cap)
            reviews.append(LotReview(
                contract_id=contract["id"],
                lot_id=lot.id,
                family=lot.family,
                available=state.available,
                verdict=risk.verdict,
                reason=reason,
                point=risk.point,
                ucb95=risk.ucb95,
                risk_limit=risk.risk_limit,
                evidence_level=risk.evidence_level,
                fused_from=risk.fused_from,
            ))
    return reviews


def _reason(state, contract: dict[str, str], risk, cap: int) -> str:
    if state.available <= 0:
        if state.consumed_recheck > 0:
            return "recheck_held"
        return "committed"
    if risk.verdict == PASS:
        if min(state.available, cap) < 1:
            return "family_cap"
        return "eligible"
    return {
        REJECT: "risk_reject",
        BORDERLINE: "borderline",
        INSUFFICIENT_EVIDENCE: "evidence_thin",
        REFERENCE_ONLY: "reference_only",
        NO_EVIDENCE: "no_evidence",
    }.get(risk.verdict, risk.verdict)


def render_reviews(reviews: list[LotReview]) -> str:
    lines = ["下一轮备货库存回看（按合同）", "-" * 50]
    for review in reviews:
        lines.append(
            f"  {review.contract_id} <- {review.lot_id}（{review.family}）"
            f"可发 {review.available}：{review.reason}，"
            f"判定 {review.verdict}，风险 {review.point:.4f}/UCB {review.ucb95:.4f}"
            f"（限值 {review.risk_limit:.2f}，证据 {review.evidence_level}）"
            + (f"，融合自 {review.fused_from}" if review.fused_from else "")
        )
    return "\n".join(lines)
