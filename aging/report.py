"""把证据、风险、复查与分配核对编排成一份可复核的报告。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .allocation import (
    evaluate_plan,
    latest_substitution_deadline,
    shipped_lot_notifications,
)
from .data import EPOCH_DATE, calendar_to_day, day_to_calendar
from .estimator import build_registry, lots_from
from .recheck import (
    available_results,
    build_plan,
    load_lab_tests,
    load_results,
    validate_windows,
)
from .risk import comparability_note, warranty_cover, window_for


# ---------------------------------------------------------------------------
# 机读结构
# ---------------------------------------------------------------------------


def _default_asof_day() -> int:
    # 资料中证据冻结日 2026-03-23 对应项目时钟 day=1。
    return calendar_to_day(EPOCH_DATE)


def run_analysis(bundle, asof_day: int | None = None, n_boot: int = 200,
                 plan_test_ids: list[list[str]] | None = None,
                 pessimistic: bool = False) -> dict[str, Any]:
    asof_day = _default_asof_day() if asof_day is None else asof_day
    asof_calendar = day_to_calendar(asof_day)
    registry = build_registry(
        bundle, n_boot=n_boot, asof=asof_calendar, pessimistic=pessimistic
    )
    lots = lots_from(bundle)
    contracts = bundle.table("contracts")
    tests = load_lab_tests(bundle)
    results = load_results(bundle)
    known = available_results(results, asof_day)

    if plan_test_ids is None:
        # 默认比较：不复查 + 每个单项复查；组合方案由调用方显式传入。
        plan_test_ids = [[]] + [[test_id] for test_id in sorted(tests)]

    plans = []
    for test_ids in plan_test_ids:
        plan = build_plan(test_ids, tests, known)
        evaluation = evaluate_plan(
            bundle, plan, contracts, lots, registry, asof_day, tests, results
        )
        plans.append(_serialize_evaluation(evaluation, tests, results, contracts))

    evidence = _evidence_summary(registry, lots, bundle)
    warranties = _warranty_summary(contracts, bundle.table("warranty"), lots)
    # 客户提示基于"截至 asof 的全部已到证据"（含迟到回执）；出库事实不撤销。
    all_known_plan = build_plan(sorted(known), tests, known)
    latest_evaluation = evaluate_plan(
        bundle, all_known_plan, contracts, lots,
        registry, asof_day, tests, results,
    )
    notifications = shipped_lot_notifications(latest_evaluation, bundle, contracts)
    deadlines = {
        contract["id"]: latest_substitution_deadline(contract, tests)
        for contract in contracts
    }
    return {
        "asof_day": asof_day,
        "asof_calendar": asof_calendar.isoformat(),
        "pessimistic_censoring": pessimistic,
        "window_validation": validate_windows(bundle, tests),
        "evidence": evidence,
        "warranty": warranties,
        "plans": plans,
        "substitution_deadlines": deadlines,
        "shipped_notifications": notifications,
        "late_results_ignored_asof": [
            test_id for test_id, result in results.items()
            if result.received_day > asof_day
        ],
    }


def _evidence_summary(registry, lots, bundle) -> list[dict[str, Any]]:
    out = []
    for lot in lots:
        fit_bundle = registry.for_lot(lot)
        direct = registry.direct(lot)
        chain = []
        for item in registry.evidence_chain(lot):
            chain.append({
                "level": item.evidence_level,
                "specimens": item.fit.n_specimens,
                "events": item.fit.n_events,
                "exp_rate": round(item.fit.rate, 5),
                "weibull_shape": round(item.weibull.shape, 3),
                "weibull_scale": round(item.weibull.scale, 2),
                "support": [item.support_from, item.support_to],
            })
        out.append({
            "lot": lot.id,
            "family": lot.family,
            "grade": lot.grade,
            "condition": lot.condition,
            "comparability": comparability_note(lot, fit_bundle, direct),
            "evidence_level": fit_bundle.evidence_level,
            "direct_specimens": direct.fit.n_specimens if direct else 0,
            "direct_events": direct.fit.n_events if direct else 0,
            "n_specimens": fit_bundle.fit.n_specimens,
            "n_events": fit_bundle.fit.n_events,
            "ess": round(fit_bundle.fit.ess, 2),
            "exp_rate": round(fit_bundle.fit.rate, 5),
            "weibull_shape_sensitivity": round(fit_bundle.weibull.shape, 3),
            "weibull_scale_sensitivity": round(fit_bundle.weibull.scale, 2),
            "support_age": [fit_bundle.support_from, fit_bundle.support_to],
            "end_reasons": fit_bundle.reason_counts,
            "bootstrap_valid": int(fit_bundle.bootstrap_rates.size),
            "evidence_chain": chain,
        })
    return out


def _warranty_summary(contracts, warranty_rows, lots) -> list[dict[str, Any]]:
    out = []
    lot_by_id = {lot.id: lot.id for lot in lots}
    for contract in contracts:
        # 质保说明与具体批无关（窗口长度只取决于合同），用任一匹配批演示龄期。
        matched = [lot for lot in lots
                   if lot.grade == contract["grade"] and lot.condition == contract["condition"]]
        if not matched:
            continue
        window = window_for(matched[0], contract)
        cover = warranty_cover(contract, warranty_rows, window)
        out.append({
            "contract": contract["id"],
            "warranty": cover.warranty_id,
            "claim_until_days": cover.claim_until,
            "duration_days": int(contract["duration"]),
            "covered_days": cover.covered_days,
            "uncovered_tail_days": cover.uncovered_days,
            "claim_end_age_example": cover.claim_end_age,
            "note": cover.note,
        })
    return out


def _serialize_evaluation(evaluation, tests, results, contracts) -> dict[str, Any]:
    return {
        "recheck_tests": evaluation.plan.test_ids,
        "consumed_by_lot": evaluation.plan.consumed_by_lot,
        "known_results_asof": sorted(evaluation.plan.results),
        "lot_availability": [
            {
                "lot": state.lot.id,
                "gross": state.gross_qty,
                "committed": state.committed,
                "recheck_consumed": state.consumed_recheck,
                "available": state.available,
            }
            for state in evaluation.states.values()
        ],
        "contracts": [
            _serialize_feasibility(evaluation.contracts[contract["id"]])
            for contract in contracts
        ],
        "segment_risks": [
            _serialize_risk(risk)
            for state in evaluation.states.values()
            for risk in state.segment_risks.values()
        ],
        "branches": {
            test_id: {
                branch: {
                    contract_id: _serialize_feasibility(feasibility)
                    for contract_id, feasibility in branch_map.items()
                }
                for branch, branch_map in branch_dict.items()
            }
            for test_id, branch_dict in evaluation.branches.items()
        },
        "coverage_warnings": evaluation.coverage_warnings,
        "double_use_conflicts": evaluation.double_use_conflicts,
    }


def _serialize_feasibility(feasibility) -> dict[str, Any]:
    return {
        "need": feasibility.need,
        "mix_allowed": feasibility.mix_allowed,
        "family_cap": feasibility.family_cap,
        "feasible": feasibility.feasible,
        "reason": feasibility.reason,
        "shortfall": feasibility.shortfall,
        "lines": [asdict(line) for line in feasibility.lines],
    }


def _serialize_risk(risk) -> dict[str, Any]:
    return {
        "lot": risk.lot_id,
        "contract": risk.contract_id,
        "age_segment": [risk.start_age, risk.end_age],
        "point": round(risk.point, 4),
        "ucb95": round(risk.ucb95, 4),
        "risk_limit": risk.risk_limit,
        "verdict": risk.verdict,
        "estimability": risk.estimability,
        "evidence_level": risk.evidence_level,
        "direct_n_events": risk.direct_n_events,
        "support_age": [risk.support_from, risk.support_to],
        "bootstrap_coverage": round(risk.bootstrap_coverage, 3),
    }


# ---------------------------------------------------------------------------
# 人读渲染
# ---------------------------------------------------------------------------

_VERDICT_CN = {
    "pass": "合格",
    "borderline": "临界(点估计达标但上界超限)",
    "reject": "不合格",
    "no_evidence": "证据不足不可估计",
    "insufficient_evidence": "当前生产线证据太薄，不能放行",
    "reference_only": "仅参考/全层证据，不能据此放行",
}

_EST_CN = {
    "estimable": "支撑内可估计",
    "extrapolated": "含龄期外推",
    "unestimable": "无法估计",
}

_LEVEL_CN = {
    "direct": "同家族直配",
    "current_pooled": "当前家族池化",
    "thin_current_pooled": "当前家族薄池化",
    "full_pooled": "全层池化(含参考线)",
    "reference_pooled": "参考家族池化",
}


def render_text(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("防护贴老化挑批核对报告")
    lines.append("=" * 60)
    lines.append(f"证据时点：项目时钟 day {analysis['asof_day']}（{analysis['asof_calendar']}）")
    if analysis.get("pessimistic_censoring"):
        lines.append("口径：悲观敏感性——售出/换件/其他退回归件按观察窗内可能失效处理")
    if analysis["late_results_ignored_asof"]:
        lines.append("以下复查回执在该时点之后到达，未用于本时点决策："
                     + ", ".join(analysis["late_results_ignored_asof"]))
    if analysis["window_validation"]:
        lines.append("复查窗口校验问题：")
        lines.extend(f"  - {item}" for item in analysis["window_validation"])
    else:
        lines.append("复查窗口容量与 lot_options 校验通过")

    lines.append("")
    lines.append("一、证据基础与可比性（主模型=指数恒定风险；Weibull 仅形状敏感性对照）")
    for item in analysis["evidence"]:
        lines.append(
            f"  {item['lot']}（{item['family']} {item['grade']}/{item['condition']}）："
            f"主模型试件 {item['n_specimens']}、失效事件 {item['n_events']}、ESS {item['ess']}；"
            f"指数日风险率 r={item['exp_rate']:.5f}；"
            f"Weibull 对照 k={item['weibull_shape_sensitivity']} "
            f"λ={item['weibull_scale_sensitivity']}；"
            f"支撑龄期 {item['support_age'][0]}–{item['support_age'][1]} 天；"
            f"证据层级【{_LEVEL_CN[item['evidence_level']]}】；"
            f"同家族自身 {item['direct_specimens']} 试件/{item['direct_events']} 事件；"
            f"终止原因 {item['end_reasons']}")
        for link in item["evidence_chain"]:
            lines.append(
                f"      证据链 {_LEVEL_CN[link['level']]}：{link['specimens']} 试件/"
                f"{link['events']} 事件，r={link['exp_rate']:.5f}，"
                f"Weibull 对照 k={link['weibull_shape']} λ={link['weibull_scale']}，"
                f"支撑 {link['support'][0]}–{link['support'][1]} 天")
        lines.append(f"      可比性：{item['comparability']}")

    lines.append("")
    lines.append("二、质保界线（只影响申诉覆盖，不改物理风险）")
    for item in analysis["warranty"]:
        lines.append(f"  {item['contract']}：{item['note']}")

    lines.append("")
    lines.append("三、各复查安排下的可行性对比")
    for index, plan in enumerate(analysis["plans"]):
        title = "不复查（基准）" if not plan["recheck_tests"] else "复查 " + "+".join(plan["recheck_tests"])
        lines.append(f"  方案 {index}: {title}")
        if plan["consumed_by_lot"]:
            lines.append("    抽离占量：" + ", ".join(
                f"{lot} {qty} 件" for lot, qty in plan["consumed_by_lot"].items()))
        for availability in plan["lot_availability"]:
            lines.append(
                f"    库存 {availability['lot']}：毛 {availability['gross']} - "
                f"承诺 {availability['committed']} - 抽离 {availability['recheck_consumed']} "
                f"= 可发 {availability['available']}")
        for feasibility in plan["contracts"]:
            mark = "可行" if feasibility["feasible"] else "不可行"
            lines.append(f"    合同可行性 {mark}：{feasibility['reason']}")
            for line in feasibility["lines"]:
                lines.append(
                    f"      分配 {line['lot_id']} {line['qty']} 件（{line['family']}），"
                    f"判定 {_VERDICT_CN[line['verdict']]}，"
                    f"点估计 {line['point']:.4f} / UCB95 {line['ucb95']:.4f}，"
                    f"{_EST_CN[line['estimability']]}")
        for risk in plan["segment_risks"]:
            lines.append(
                f"    段风险 {risk['lot']}->{risk['contract']} "
                f"龄期 {risk['age_segment'][0]}–{risk['age_segment'][1]}："
                f"{risk['point']:.4f}（UCB95 {risk['ucb95']:.4f}，限值 {risk['risk_limit']:.2f}）"
                f" -> {_VERDICT_CN[risk['verdict']]} / {_EST_CN[risk['estimability']]}"
                f" / {_LEVEL_CN[risk['evidence_level']]}（同家族事件 {risk['direct_n_events']}）")
        for test_id, branch_dict in plan["branches"].items():
            for branch, branch_map in branch_dict.items():
                feasible = [f"{cid}:{'可行' if f['feasible'] else '不可行'}"
                            for cid, f in branch_map.items()]
                lines.append(f"    未观测分支 {test_id}/{branch}：{', '.join(feasible)}")
        for warning in plan["coverage_warnings"]:
            lines.append(f"    覆盖提醒：{warning}")
        for conflict in plan["double_use_conflicts"]:
            lines.append(f"    重复占用：{conflict}")

    lines.append("")
    lines.append("四、替代决定截止时点（None=回执赶不上该合同发运）")
    for contract_id, deadlines in analysis["substitution_deadlines"].items():
        rendered = ", ".join(
            f"{test_id}: {deadline if deadline is not None else '不可用'}"
            for test_id, deadline in deadlines.items())
        lines.append(f"  {contract_id}：{rendered}")

    lines.append("")
    lines.append("五、已出库批的客户提示（不撤销出库事实）")
    if not analysis["shipped_notifications"]:
        lines.append("  无")
    for item in analysis["shipped_notifications"]:
        source = f"，证据更新自复查 {item['fused_from']}" if item["fused_from"] else ""
        lines.append(
            f"  {item['contract']} / {item['lot']} {item['qty']} 件："
            f"{_VERDICT_CN[item['verdict']]}，点估计 {item['point']} / "
            f"UCB95 {item['ucb95']}（限值 {item['limit']}）{source}。{item['action']}")
        if item.get("late_timing"):
            lines.append(f"      时点纪律：{item['late_timing']}")
    lines.append("=" * 60)
    return "\n".join(lines)
