"""生存估计核心的不变量测试。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from aging.data import Observation, load_bundle
from aging.survival import (
    build_intervals,
    fit_exponential,
    fit_turnbull,
    fit_weibull,
    normalized_weights,
)
from datetime import date


def _obs(specimen: str, entry: int, last_ok: int, first_failed, last_seen: int,
         reason: str = "failure", prob: float = 0.5, seen: str = "2026-03-01") -> Observation:
    return Observation(
        specimen=specimen, family="FX", grade="A", condition="normal",
        entry_age=entry, last_ok=last_ok, first_failed=first_failed,
        last_seen=last_seen, end_reason=reason, seen_on=date.fromisoformat(seen),
        sample_prob=prob,
    )


class IntervalTests(unittest.TestCase):
    def test_failure_is_interval_censored_not_point(self):
        obs = _obs("X", 10, 30, 40, 40)
        intervals = build_intervals([obs])
        self.assertEqual(intervals[0].left, 30)
        self.assertEqual(intervals[0].right, 40)
        self.assertEqual(intervals[0].entry, 10)

    def test_censoring_reasons_kept_distinct_and_right_censored(self):
        for reason in ("sold", "replaced", "returned_other", "monitoring"):
            obs = _obs("X", 5, 20, None, 22, reason=reason)
            intervals = build_intervals([obs])
            self.assertIsNone(intervals[0].right)
            self.assertEqual(intervals[0].reason, reason)

    def test_pessimistic_mode_treats_termination_as_possible_failure(self):
        obs = _obs("X", 5, 20, None, 22, reason="sold")
        base = build_intervals([obs])
        pessimistic = build_intervals([obs], pessimistic=True)
        self.assertIsNone(base[0].right)
        self.assertEqual(pessimistic[0].right, 22)
        self.assertEqual(pessimistic[0].left, 20)

    def test_inverse_probability_weights_ess(self):
        obs = [_obs(f"S{i}", 5, 10 + i, 20 + i, 20 + i, prob=p)
               for i, p in enumerate((0.25, 0.5, 1.0))]
        weights = normalized_weights(build_intervals(obs))
        raw = np.array([1 / 0.25, 1 / 0.5, 1 / 1.0])
        expected_ess = raw.sum() ** 2 / np.square(raw).sum()
        self.assertAlmostEqual(float(weights.sum()), expected_ess, places=6)


class ExponentialFitTests(unittest.TestCase):
    def setUp(self):
        # 构造恒定风险 r=0.02 的合成区间数据。
        rng = np.random.default_rng(7)
        self.observations = []
        rate = 0.02
        for i in range(60):
            entry = int(rng.integers(2, 15))
            lifetime = int(rng.exponential(1 / rate))
            last_seen = entry + int(rng.integers(10, 40))
            if entry + lifetime <= last_seen:
                first_failed = entry + lifetime
                last_ok = first_failed - int(rng.integers(3, 10))
                self.observations.append(_obs(
                    f"F{i}", entry, max(entry, last_ok), first_failed,
                    max(first_failed, last_seen)))
            else:
                self.observations.append(_obs(
                    f"C{i}", entry, last_seen, None, last_seen, reason="monitoring"))

    def test_recovers_known_rate(self):
        intervals = build_intervals(self.observations)
        fit = fit_exponential(intervals)
        # 60 个试件应把速率估计拉回真值的 ±35% 区间。
        self.assertAlmostEqual(fit.rate, 0.02, delta=0.007)
        self.assertEqual(fit.n_events, sum(i.right is not None for i in intervals))

    def test_monotone_survival_and_valid_probabilities(self):
        fit = fit_exponential(build_intervals(self.observations))
        ages = np.linspace(0, 100, 11)
        surv = fit.survival(ages)
        self.assertTrue(np.all(np.diff(surv) <= 1e-12))
        self.assertTrue(np.all((surv >= 0) & (surv <= 1)))

    def test_no_extreme_rate_on_tiny_interval_data(self):
        # 回归：近零概率差被夹成 0 时，优化器不得逃到极端速率。
        obs = [
            _obs("A", 14, 31, 45, 45, prob=0.65),
            _obs("B", 15, 32, 56, 56, prob=0.6),
            _obs("C", 22, 38, None, 38, reason="replaced", prob=0.44),
            _obs("D", 20, 33, 59, 59, prob=0.44),
        ]
        fit = fit_exponential(build_intervals(obs))
        self.assertTrue(0.001 < fit.rate < 0.2)

    def test_turnbull_mass_sums_to_one_and_respects_grid(self):
        fit = fit_turnbull(build_intervals(self.observations))
        self.assertAlmostEqual(float(fit.mass.sum()), 1.0, places=6)
        self.assertTrue(np.all(fit.mass >= -1e-12))

    def test_weibull_finite_on_tiny_data(self):
        obs = [_obs("A", 14, 31, 45, 45), _obs("B", 22, 38, None, 38, reason="sold")]
        fit = fit_weibull(build_intervals(obs))
        self.assertTrue(math.isfinite(fit.shape) and math.isfinite(fit.scale))


class BundleLoadingTests(unittest.TestCase):
    def test_latest_observation_replaces_older_row(self):
        bundle = load_bundle("data")
        records = {o.specimen: o for o in bundle.observations()}
        # S04 有两行，2026-04-02 的失效行必须替代 2026-03-12 的删失行。
        self.assertEqual(records["S04"].end_reason, "failure")
        self.assertEqual(records["S04"].first_failed, 56)

    def test_asof_filter_excludes_future_evidence(self):
        bundle = load_bundle("data")
        asof = date(2026, 3, 20)
        records = bundle.observations_asof(asof)
        self.assertTrue(all(o.seen_on <= asof for o in records))
        specimen_rows = {o.specimen: o for o in records}
        # 截至 3 月 20 日，S04 的失效更新（4 月）尚不可见，仍是 sold 删失。
        self.assertEqual(specimen_rows["S04"].end_reason, "sold")


if __name__ == "__main__":
    unittest.main()
