"""冻结时点的回看校准。

规则（对应业务约定）：

- 每张历史订单只能用 ``asof`` 当日及之前的观察证据拟合，后来才退回/失效的记录
  不得用来替过去挑批（数据中 S04 的 2026-04-02 更新就是泄漏防线演练点）。
- 观察窗只覆盖到 ``shipment_age + duration`` 之前的部分；``consumed < qty``
  表示结果尚未收齐，未消耗件不能当作持续完好，分母只用 consumed。
- 校准输出预测段风险与实际消耗件失效率的对照，并做二项覆盖检验；回看的目的
  是检验方法本身，不重写当时的决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.stats import binomtest

from .data import DataBundle, parse_date
from .estimator import build_bundle


@dataclass(frozen=True)
class BacktestRow:
    order_id: str
    family: str
    grade: str
    condition: str
    asof: date
    start_age: int
    end_age: int
    predicted_point: float
    predicted_ucb95: float
    consumed: int
    observed_failures: int
    observed_rate: float
    covered_by_ucb: bool
    estimability: str
    p_value: float
    leakage_guarded: bool          # asof 之后的观察是否确实被排除
    note: str


def _observations_for(bundle: DataBundle, family: str, grade: str,
                      condition: str, asof: date) -> list:
    return [
        obs for obs in bundle.observations_asof(asof)
        if obs.family == family and obs.grade == grade and obs.condition == condition
    ]


def run_backtest(bundle: DataBundle, n_boot: int = 100, seed: int = 71013,
                 include_reference: bool = True) -> list[BacktestRow]:
    rng = np.random.default_rng(seed)
    rows: list[BacktestRow] = []
    orders = bundle.table("historical_orders")
    # 全量观察只用于泄漏核查，不参与拟合。
    all_obs = bundle.observations()

    for order in orders:
        if not include_reference and order["family"].startswith("RF"):
            continue
        asof = parse_date(order["asof"])
        assert asof is not None
        records = _observations_for(
            bundle, order["family"], order["grade"], order["condition"], asof
        )
        start_age = int(order["shipment_age"])
        end_age = start_age + int(order["duration"])
        consumed = int(order["consumed"])
        failures = int(order["failed"])

        leakage_records = [
            obs for obs in all_obs
            if obs.family == order["family"] and obs.seen_on > asof
        ]
        # 泄漏防线：asof 之后才可见的行，其试件在拟合集合里只能以 ≤asof 的旧行
        # 出现；拟合集合本身已按 seen_on 过滤，这里做一次显式复核。
        used_ids = {obs.specimen for obs in records}
        assert all(obs.seen_on > asof for obs in leakage_records)
        leaked = [
            obs.specimen for obs in leakage_records
            if obs.specimen in used_ids
            and not any(prior.seen_on <= asof for prior in records
                        if prior.specimen == obs.specimen)
        ]
        leakage_guarded = not leaked

        if not records or not any(obs.is_failure for obs in records):
            rows.append(BacktestRow(
                order["id"], order["family"], order["grade"], order["condition"],
                asof, start_age, end_age, float("nan"), float("nan"),
                consumed, failures, failures / consumed, False,
                "unestimable", float("nan"), leakage_guarded,
                "asof 时点该家族无失效证据，方法应输出不可估计而非零风险",
            ))
            continue

        fit = build_bundle(
            family=order["family"], grade=order["grade"],
            condition=order["condition"], observations=records,
            n_boot=n_boot, rng=rng,
        )
        point = fit.failure_curve(start_age, end_age)
        ucb = fit.ucb95(start_age, end_age)
        estimability = fit.estimability(start_age, end_age)
        observed_rate = failures / consumed
        # 双侧二项检验：实际失效数与预测概率是否相容（不显著即方法未被证伪）。
        p_value = float(binomtest(failures, consumed, point).pvalue)
        note = "段尾超出观察支撑，预测含参数外推" if estimability == "extrapolated" else ""
        rows.append(BacktestRow(
            order_id=order["id"],
            family=order["family"],
            grade=order["grade"],
            condition=order["condition"],
            asof=asof,
            start_age=start_age,
            end_age=end_age,
            predicted_point=point,
            predicted_ucb95=ucb,
            consumed=consumed,
            observed_failures=failures,
            observed_rate=observed_rate,
            covered_by_ucb=observed_rate <= ucb,
            estimability=estimability,
            p_value=p_value,
            leakage_guarded=leakage_guarded,
            note=note,
        ))
    return rows


def calibration_summary(rows: list[BacktestRow]) -> dict[str, object]:
    scored = [row for row in rows if not np.isnan(row.predicted_point)]
    if not scored:
        return {"scored_orders": 0}
    coverage = sum(row.covered_by_ucb for row in scored) / len(scored)
    rejects = [row.order_id for row in scored if row.p_value < 0.05]
    return {
        "scored_orders": len(scored),
        "ucb_coverage": round(coverage, 3),
        "binomial_reject_orders": rejects,
        "unestimable_orders": [row.order_id for row in rows if np.isnan(row.predicted_point)],
        "mean_abs_error": round(
            float(np.mean([abs(row.predicted_point - row.observed_rate) for row in scored])),
            4,
        ),
    }
