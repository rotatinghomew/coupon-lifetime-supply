"""命令行入口：输出人读报告与机读 JSON。

用法：
    python3 run_analysis.py                     # 默认证据时点、默认复查方案集
    python3 run_analysis.py --asof-day 3        # 指定项目时钟时点
    python3 run_analysis.py --plans T1 T2       # 比较指定复查组合
    python3 run_analysis.py --backtest          # 追加冻结时点回看校准
    python3 run_analysis.py --json out.json     # 同时落机读结果
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from aging.backtest import calibration_summary, run_backtest
from aging.data import load_bundle
from aging.report import render_text, run_analysis


def _to_jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="防护贴老化挑批核对")
    parser.add_argument("directory", nargs="?", default="data")
    parser.add_argument("--asof-day", type=int, default=None)
    parser.add_argument("--boot", type=int, default=200)
    parser.add_argument("--plans", nargs="*", action="append", default=None)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument(
        "--pessimistic", action="store_true",
        help="把售出/换件/其他退回的终止件视作观察窗内可能失效的敏感性口径",
    )
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument(
        "--review", action="store_true",
        help="输出下一轮备货的合同视角库存回看（可选性与不可选原因）",
    )
    args = parser.parse_args()

    bundle = load_bundle(Path(args.directory))
    plan_sets = None
    if args.plans:
        plan_sets = [group for group in args.plans]
    analysis = run_analysis(
        bundle, asof_day=args.asof_day, n_boot=args.boot,
        plan_test_ids=plan_sets, pessimistic=args.pessimistic,
    )
    print(render_text(analysis))

    if args.review:
        from aging.review import render_reviews, review_inventory
        reviews = review_inventory(
            bundle, asof_day=args.asof_day or 1,
            plan_test_ids=[group for group in (args.plans or [])][0] if args.plans else None,
            n_boot=args.boot,
        )
        print("\n" + render_reviews(reviews))
        analysis["inventory_review"] = [_to_jsonable(item) for item in reviews]

    if args.backtest:
        rows = run_backtest(bundle, n_boot=max(50, args.boot // 2))
        print("\n冻结时点回看校准")
        print("-" * 60)
        for row in rows:
            if row.estimability == "unestimable":
                print(f"  {row.order_id} {row.family}: {row.note}")
                continue
            print(
                f"  {row.order_id} {row.family} 龄期 {row.start_age}-{row.end_age}："
                f"预测 {row.predicted_point:.3f} (UCB {row.predicted_ucb95:.3f}) "
                f"vs 实际 {row.observed_rate:.3f}（{row.observed_failures}/{row.consumed}），"
                f"UCB 覆盖={row.covered_by_ucb}，p={row.p_value:.3f} {row.note}")
        print("  汇总：", json.dumps(calibration_summary(rows), ensure_ascii=False))
        analysis["backtest"] = {
            "rows": [_to_jsonable(row) for row in rows],
            "summary": calibration_summary(rows),
        }

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(_to_jsonable(analysis), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n机读结果已写入 {args.json_path}")


if __name__ == "__main__":
    main()
