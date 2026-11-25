"""风险判定、证据层级、复查融合与分配约束的集成测试。"""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np

from aging.allocation import (
    evaluate_plan,
    latest_substitution_deadline,
    shipped_lot_notifications,
)
from aging.data import load_bundle
from aging.estimator import build_registry, lots_from
from aging.recheck import build_plan, load_lab_tests, load_results
from aging.risk import (
    BORDERLINE,
    INSUFFICIENT_EVIDENCE,
    NO_EVIDENCE,
    PASS,
    REJECT,
    REFERENCE_ONLY,
)


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle("data")
        cls.registry = build_registry(cls.bundle, n_boot=40, seed=11)
        cls.lots = lots_from(cls.bundle)

    def test_thin_current_families_not_released_on_pool_alone(self):
        # F01–F05 自身只有 1–2 个试件，主模型必须落在薄池化或更保守层级。
        for lot in self.lots:
            fit_bundle = self.registry.for_lot(lot)
            self.assertIn(fit_bundle.evidence_level,
                          ("thin_current_pooled", "full_pooled", "reference_pooled"))

    def test_conditions_never_mixed_across_strata(self):
        # humid 批永远拿不到 normal 层的模型，反之亦然。
        for lot in self.lots:
            fit_bundle = self.registry.for_lot(lot)
            self.assertEqual(fit_bundle.grade, lot.grade)
            self.assertEqual(fit_bundle.condition, lot.condition)

    def test_evidence_chain_ordered_direct_to_reference(self):
        chain = self.registry.evidence_chain(self.lots[0])
        levels = [item.evidence_level for item in chain]
        self.assertEqual(levels[0], "direct")
        self.assertIn("full_pooled", levels)
        self.assertIn("reference_pooled", levels)

    def test_ucb_is_never_below_point_estimate(self):
        contracts = self.bundle.table("contracts")
        for lot in self.lots:
            fit_bundle = self.registry.for_lot(lot)
            for contract in contracts:
                if (lot.grade, lot.condition) != (contract["grade"], contract["condition"]):
                    continue
                from aging.risk import assess_segment, window_for
                risk = assess_segment(fit_bundle, lot, contract,
                                      direct_bundle=self.registry.direct(lot))
                self.assertGreaterEqual(risk.ucb95 + 1e-9, risk.point)
                self.assertIn(risk.verdict,
                              (PASS, BORDERLINE, REJECT, NO_EVIDENCE,
                               INSUFFICIENT_EVIDENCE, REFERENCE_ONLY))


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle("data")
        cls.registry = build_registry(cls.bundle, n_boot=40, seed=23)
        cls.lots = lots_from(cls.bundle)
        cls.contracts = cls.bundle.table("contracts")
        cls.tests = load_lab_tests(cls.bundle)
        cls.results = load_results(cls.bundle)

    def _evaluate(self, test_ids, asof_day):
        plan = build_plan(test_ids, self.tests,
                          {tid: r for tid, r in self.results.items()
                           if r.received_day <= asof_day})
        return evaluate_plan(
            self.bundle, plan, self.contracts, self.lots,
            self.registry, asof_day, self.tests, self.results,
        )

    def test_recheck_consumption_reduces_available_stock(self):
        baseline = self._evaluate([], 1)
        with_t1 = self._evaluate(["T1"], 1)
        self.assertEqual(baseline.states["I01"].available, 11)   # 39-28
        self.assertEqual(with_t1.states["I01"].available, 5)     # 再抽 6

    def test_future_results_excluded_at_early_asof(self):
        early = self._evaluate(["T1", "T3"], 1)
        self.assertEqual(early.plan.results, {})
        # day 1 时 T1/T3 都只是分支，不能产生融合判定。
        self.assertTrue(all(not state.fused for state in early.states.values()))

    def test_fusion_only_inside_tested_age_window(self):
        # T1 只检查到 34 天，C01/C02 窗口都从 38 天起 -> 不融合，只提醒。
        day3 = self._evaluate(["T1"], 3)
        self.assertIn("T1", day3.states["I01"].fused)
        risk = day3.states["I01"].segment_risks["C02"]
        self.assertIsNone(risk.fused_from)
        self.assertTrue(any("T1" in warning and "超出" in warning
                            for warning in day3.coverage_warnings))

    def test_fusion_applies_when_window_covered(self):
        # T3 检查到 72 天，覆盖 C03 的 50–69 窗口；day 7 回执已到。
        day7 = self._evaluate(["T3"], 7)
        risk = day7.states["I03"].segment_risks["C03"]
        self.assertEqual(risk.fused_from, "T3")
        self.assertEqual(risk.verdict, REJECT)  # 3/7 失效，远超 0.15 限值

    def test_unobserved_branch_cannot_guarantee_two_contracts(self):
        evaluation = self._evaluate(["T4"], 1)
        self.assertIn("T4", evaluation.branches)
        for branch_map in evaluation.branches.values():
            self.assertIn("pass_branch", branch_map)
            self.assertIn("fail_branch", branch_map)

    def test_substitution_deadline_marks_unreachable_results(self):
        c02 = next(c for c in self.contracts if c["id"] == "C02")
        deadlines = latest_substitution_deadline(c02, self.tests)
        # C02 day 4 发运；T2 day 5 才返回 -> 不可用。
        self.assertIsNone(deadlines["T2"])
        self.assertEqual(deadlines["T1"], 3)

    def test_shipped_notification_preserves_shipment_fact(self):
        day7 = self._evaluate(["T3"], 7)
        notes = shipped_lot_notifications(day7, self.bundle, self.contracts)
        c03_notes = [n for n in notes if n["contract"] == "C03"]
        self.assertEqual(len(c03_notes), 1)
        self.assertIn("不撤销", c03_notes[0]["action"])
        self.assertIn("晚于", c03_notes[0]["late_timing"])

    def test_mix_disallowed_requires_single_lot(self):
        c02 = next(c for c in self.contracts if c["id"] == "C02")
        evaluation = self._evaluate([], 1)
        feasibility = evaluation.contracts["C02"]
        self.assertFalse(feasibility.feasible)
        self.assertFalse(feasibility.mix_allowed)

    def test_confirmed_but_unshipped_pick_is_binding(self):
        # P01 确认占用 I01 28 件，即使未发运也必须计入承诺。
        evaluation = self._evaluate([], 1)
        self.assertEqual(evaluation.states["I01"].committed, 28)


class WarrantySeparationTests(unittest.TestCase):
    def test_warranty_does_not_enter_physical_risk(self):
        from aging.risk import assess_segment, warranty_cover, window_for
        bundle = load_bundle("data")
        registry = build_registry(bundle, n_boot=30, seed=5)
        lot = next(lot for lot in lots_from(bundle) if lot.id == "I01")
        contract = next(c for c in bundle.table("contracts") if c["id"] == "C01")
        fit_bundle = registry.for_lot(lot)
        risk = assess_segment(fit_bundle, lot, contract,
                              direct_bundle=registry.direct(lot))
        window = window_for(lot, contract)
        cover = warranty_cover(contract, bundle.table("warranty"), window)
        # 物理段风险不依赖保修条款；保修只给出覆盖天数。
        self.assertIsInstance(risk.point, float)
        self.assertEqual(cover.covered_days + cover.uncovered_days,
                         int(contract["duration"]))


if __name__ == "__main__":
    unittest.main()
