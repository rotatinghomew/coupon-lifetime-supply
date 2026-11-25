"""从观察记录装配可复用的拟合注册表（证据分层）。

证据纪律（对应业务上不同生产线基线不同的事实）：

1. ``direct``   库存所属家族自身的观察。事件太少时不足以支撑 Weibull 尾部，
   只能报告极宽的不确定性，不能给出"合格"判定。
2. ``current``  同 (grade, condition) 层内、当前生产家族（F*）的池化证据。
   生产条件相近，可作为主模型，但必须标注为池化并保留分层自助法。
3. ``reference`` 只剩参考家族（RF*）同层证据。它用于方法校准与可比性参照，
   不能替代当前生产线证据做合同合格判定。
4. 连参考层都没有 -> 不可估计，禁止用零风险填充。

humid 与 normal 等不同使用条件永远不进入同一条曲线。
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .data import DataBundle, Observation, group_by
from .risk import FitRegistry, Lot
from .survival import (
    CURRENT_POOLED,
    DIRECT,
    FULL_POOLED,
    REFERENCE_POOLED,
    THIN_CURRENT_POOL,
    FitBundle,
    build_intervals,
    family_stratified_bootstrap,
    fit_exponential,
    fit_turnbull,
    fit_weibull,
    normalized_weights,
)

# 单家族至少要有这么多区间删失事件/试件，直配模型才允许给出 pass 判定。
MIN_DIRECT_EVENTS = 3
MIN_DIRECT_SPECIMENS = 5
# 当前家族同层池化的最低试件量；低于它，池化模型标记为 thin，不能单独放行。
MIN_POOL_SPECIMENS = 8


def build_bundle(family: str, grade: str, condition: str,
                 observations: list[Observation], n_boot: int,
                 rng: np.random.Generator, pessimistic: bool = False,
                 evidence_level: str = DIRECT,
                 fallback_family: str | None = None) -> FitBundle:
    intervals = build_intervals(observations, pessimistic=pessimistic)
    weights = normalized_weights(intervals)
    exponential = fit_exponential(intervals, weights)
    weibull = fit_weibull(intervals, weights)
    turnbull = fit_turnbull(intervals, weights)
    # 自助路径只拟合主模型（指数）；Weibull 形状对照仅需点估计一次。
    rates, shapes, scales, _valid = family_stratified_bootstrap(
        observations, n_boot=n_boot, rng=rng,
        pessimistic=pessimistic, include_weibull=False,
    )
    reason_counts: dict[str, int] = {}
    for obs in observations:
        reason_counts[obs.end_reason] = reason_counts.get(obs.end_reason, 0) + 1
    return FitBundle(
        family=family,
        grade=grade,
        condition=condition,
        fit=exponential,
        weibull=weibull,
        turnbull=turnbull,
        support_from=turnbull.observed_from,
        support_to=turnbull.observed_to,
        reason_counts=reason_counts,
        sample_probs=[obs.sample_prob for obs in observations],
        bootstrap_rates=rates,
        bootstrap_shapes=shapes,
        bootstrap_scales=scales,
        fallback_family=fallback_family,
        evidence_level=evidence_level,
    )


def _is_current(family: str) -> bool:
    return family.startswith("F") and not family.startswith("RF")


def build_registry(bundle: DataBundle, n_boot: int = 200, seed: int = 20261125,
                   asof: date | None = None,
                   pessimistic: bool = False) -> FitRegistry:
    rng = np.random.default_rng(seed)
    observations = bundle.observations_asof(asof)

    family_bundles: dict[tuple[str, str, str], FitBundle] = {}
    current_pools: dict[tuple[str, str], FitBundle] = {}
    full_pools: dict[tuple[str, str], FitBundle] = {}
    reference_pools: dict[tuple[str, str], FitBundle] = {}

    by_family = group_by(
        observations, lambda obs: (obs.family, obs.grade, obs.condition)
    )
    for (family, grade, condition), records in sorted(by_family.items()):
        family_bundles[(family, grade, condition)] = build_bundle(
            family, grade, condition, records, n_boot=n_boot, rng=rng,
            pessimistic=pessimistic, evidence_level=DIRECT,
        )

    by_stratum = group_by(observations, lambda obs: obs.stratum)
    for (grade, condition), records in sorted(by_stratum.items()):
        current_records = [obs for obs in records if _is_current(obs.family)]
        reference_records = [obs for obs in records if not _is_current(obs.family)]
        if current_records:
            thin = len(current_records) < MIN_POOL_SPECIMENS
            current_pools[(grade, condition)] = build_bundle(
                family="CURRENT_POOL", grade=grade, condition=condition,
                observations=current_records, n_boot=n_boot, rng=rng,
                pessimistic=pessimistic,
                evidence_level=THIN_CURRENT_POOL if thin else CURRENT_POOLED,
                fallback_family="CURRENT_POOL",
            )
        # 全层池化：当前+参考同条件，信息量大，但生产线基线不同，只能作保守参照。
        full_pools[(grade, condition)] = build_bundle(
            family="FULL_POOL", grade=grade, condition=condition,
            observations=records, n_boot=n_boot, rng=rng,
            pessimistic=pessimistic, evidence_level=FULL_POOLED,
            fallback_family="FULL_POOL",
        )
        if reference_records:
            reference_pools[(grade, condition)] = build_bundle(
                family="REFERENCE_POOL", grade=grade, condition=condition,
                observations=reference_records, n_boot=n_boot, rng=rng,
                pessimistic=pessimistic, evidence_level=REFERENCE_POOLED,
                fallback_family="REFERENCE_POOL",
            )
    return FitRegistry(family_bundles, current_pools, full_pools, reference_pools)


def lots_from(bundle: DataBundle) -> list[Lot]:
    return [Lot.from_row(row) for row in bundle.table("inventory")]
