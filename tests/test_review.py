"""下一轮备货库存回看的原因分类测试。"""

from __future__ import annotations

import unittest

from aging.data import load_bundle
from aging.review import review_inventory


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle("data")

    def test_every_matching_lot_reviewed_with_exclusive_reason(self):
        reviews = review_inventory(self.bundle, asof_day=1, n_boot=30)
        # 4 合同 × 匹配批：C01/C02 各 3 个 A/normal，C03 两个 A/humid，C04 一个 B/normal。
        self.assertEqual(len(reviews), 3 + 3 + 2 + 1)
        valid_reasons = {
            "eligible", "committed", "recheck_held", "evidence_thin",
            "reference_only", "no_evidence", "risk_reject",
            "borderline", "family_cap",
        }
        self.assertTrue({review.reason for review in reviews} <= valid_reasons)

    def test_recheck_consumption_marks_lot_unavailable(self):
        # T1 抽离 I01 6 件后，I01 对 C01 仍有 5 件可发（39-28-6），原因不再是 committed。
        reviews = review_inventory(
            self.bundle, asof_day=3, plan_test_ids=["T1"], n_boot=30)
        i01_c01 = next(r for r in reviews
                       if r.contract_id == "C01" and r.lot_id == "I01")
        self.assertEqual(i01_c01.available, 5)

    def test_late_recheck_result_fuses_only_after_arrival(self):
        before = review_inventory(
            self.bundle, asof_day=5, plan_test_ids=["T3"], n_boot=30)
        after = review_inventory(
            self.bundle, asof_day=7, plan_test_ids=["T3"], n_boot=30)
        i03_before = next(r for r in before
                          if r.contract_id == "C03" and r.lot_id == "I03")
        i03_after = next(r for r in after
                         if r.contract_id == "C03" and r.lot_id == "I03")
        self.assertIsNone(i03_before.fused_from)
        self.assertEqual(i03_after.fused_from, "T3")
        self.assertEqual(i03_after.reason, "risk_reject")


if __name__ == "__main__":
    unittest.main()
