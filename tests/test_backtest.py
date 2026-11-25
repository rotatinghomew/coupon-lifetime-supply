"""冻结时点回看的纪律测试：禁止后见之明、未收齐结果不当完好。"""

from __future__ import annotations

import unittest
from datetime import date

from aging.backtest import calibration_summary, run_backtest
from aging.data import load_bundle


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle("data")
        cls.rows = run_backtest(cls.bundle, n_boot=40)

    def test_every_historical_order_scored_or_marked_unestimable(self):
        order_ids = {row["id"] for row in self.bundle.table("historical_orders")}
        scored = {row.order_id for row in self.rows}
        self.assertEqual(order_ids, scored)

    def test_s04_april_update_leaks_into_no_march_cutoff(self):
        # H2 的 asof=2026-03-13；S04 的 2026-04-02 失效更新不得进入该时点拟合。
        h2 = next(row for row in self.rows if row.order_id == "H2")
        self.assertTrue(h2.leakage_guarded)
        # 直接核查：3 月 13 日时点 S04 仍是 sold 删失（first_failed 为空）。
        asof = date(2026, 3, 13)
        visible = {o.specimen: o for o in self.bundle.observations_asof(asof)}
        self.assertIsNone(visible["S04"].first_failed)

    def test_uncollected_results_not_counted_as_intact(self):
        # 历史订单 consumed < qty 时，失效率分母只用 consumed，不用 qty。
        for row in self.rows:
            order = next(o for o in self.bundle.table("historical_orders")
                         if o["id"] == row.order_id)
            if int(order["consumed"]) < int(order["qty"]):
                self.assertEqual(row.consumed, int(order["consumed"]))
                self.assertLessEqual(row.observed_failures, row.consumed)

    def test_ucb_covers_majority_of_realized_rates(self):
        # 方法校准目标：UCB95 的经验覆盖率应接近或高于名义水平。
        summary = calibration_summary(self.rows)
        self.assertGreaterEqual(summary["ucb_coverage"], 0.90)

    def test_predictions_are_valid_probabilities(self):
        import math
        for row in self.rows:
            if not math.isnan(row.predicted_point):
                self.assertGreaterEqual(row.predicted_point, 0.0)
                self.assertLessEqual(row.predicted_point, 1.0)
                self.assertGreaterEqual(row.predicted_ucb95, row.predicted_point - 1e-9)


if __name__ == "__main__":
    unittest.main()
