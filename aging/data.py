"""资料读入与时点治理。

关键纪律（对应业务约定）：

1. 同一试件可能出现多行，``seen_on`` 较晚的行整体替代旧行，不做逐字段合并。
2. 一切证据都有可用日期：``asof`` 之后才出现的观察、复查回执在该时点不可使用，
   必须排除——回看与计划只能基于当时已有的证据。
3. 空字段表示"未填/不适用"，绝不解释成数值零。
4. 日期换算只用于展示与质保日历核对；所有生存分析在"贴片年龄（天）"轴上进行。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

# 交换资料中项目时钟 day=1 对应的日历日。资料里证据日期均围绕该锚点。
EPOCH_DATE = date(2026, 3, 23)

# end_reason 的语义分组。failure 是事件；其余都是观察终止（右删失），
# 但终止机制不同，不能一概当同一类。
FAILURE = "failure"
CENSOR_REASONS = frozenset(
    {"sold", "replaced", "returned_other", "monitoring"}
)
# 悲观敏感性分析中，视作"可能已经失效"的终止原因（售出/换件/其他退回）。
PESSIMISTIC_REASONS = frozenset({"sold", "replaced", "returned_other"})


def parse_day(value: str) -> int | None:
    """项目时钟上的天（整数）；空串返回 None。"""
    if value is None or value == "":
        return None
    return int(value)


def parse_date(value: str) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def day_to_calendar(day: int | None) -> date | None:
    """项目时钟天 -> 日历日（day=1 即 EPOCH_DATE）。"""
    if day is None:
        return None
    return EPOCH_DATE.fromordinal(EPOCH_DATE.toordinal() + day - 1)


def calendar_to_day(day_date: date | str | None) -> int | None:
    """日历日 -> 项目时钟天；空输入返回 None。"""
    if day_date is None or day_date == "":
        return None
    if isinstance(day_date, str):
        day_date = date.fromisoformat(day_date)
    return day_date.toordinal() - EPOCH_DATE.toordinal() + 1


def parse_bool01(value: str) -> bool:
    return value == "1"


@dataclass(frozen=True)
class Observation:
    """一条已按试件去重、并可能按时点过滤后的观察记录。"""

    specimen: str
    family: str
    grade: str
    condition: str
    entry_age: int          # 左截断点：该试件进入观察时的贴片年龄
    last_ok: int            # 最后确证正常的年龄
    first_failed: int | None  # 首次确证失效年龄；空=未观察到失效
    last_seen: int          # 最后能确认状态的年龄
    end_reason: str
    seen_on: date
    sample_prob: float
    seen_day: int | None = field(default=None)  # seen_on 换算到项目时钟

    @property
    def stratum(self) -> tuple[str, str]:
        return (self.grade, self.condition)

    @property
    def is_failure(self) -> bool:
        return self.end_reason == FAILURE and self.first_failed is not None


@dataclass
class DataBundle:
    """全部交换表的强类型视图，保留原始字符串字段以便追溯。"""

    base: Path
    raw: dict[str, list[dict[str, str]]]

    def table(self, name: str) -> list[dict[str, str]]:
        return self.raw[name]

    def inventory_row(self, lot: str) -> dict[str, str]:
        for row in self.raw["inventory"]:
            if row["id"] == lot:
                return row
        raise KeyError(lot)

    # -- 观察记录的两个治理入口 -------------------------------------------------

    def observations(self) -> list[Observation]:
        """同一试件以 seen_on 最新的行替代旧行（不按时点过滤）。

        同一 seen_on 出现多行时保留交换文件中的第一行（稳定），不让后出现的
        行覆盖先出现的行。
        """
        latest: dict[str, Observation] = {}
        for obs in self.observations_in_order():
            old = latest.get(obs.specimen)
            if old is None or obs.seen_on > old.seen_on:
                latest[obs.specimen] = obs
        return sorted(latest.values(), key=lambda item: item.specimen)

    def observations_in_order(self) -> list[Observation]:
        """按交换文件顺序返回全部观察（含同一试件的历史行），不去重。"""
        return [_to_observation(row) for row in self.raw["observations"]]

    def observations_asof(self, asof: date | None = None) -> list[Observation]:
        """截至 ``asof``（含当日）可用的观察记录；asof=None 表示不裁剪。

        必须先按 seen_on 过滤、再在窗口内去重：若先全局去重，试件在 asof 之后
        才更新的行会把它在 asof 当时可见的旧行一起抹掉，造成过去时点"凭空丢
        试件"。
        """
        records = self.observations_in_order()
        if asof is not None:
            records = [record for record in records if record.seen_on <= asof]
        latest: dict[str, Observation] = {}
        for obs in records:
            old = latest.get(obs.specimen)
            if old is None or obs.seen_on > old.seen_on:
                latest[obs.specimen] = obs
        return sorted(latest.values(), key=lambda item: item.specimen)

    def supplemental_asof(self, asof_day: int | None = None) -> list[dict[str, str]]:
        """截至项目时钟 ``asof_day``（含）已到达的实际复查回执。"""
        rows = self.raw["supplemental"]
        if asof_day is None:
            return list(rows)
        return [row for row in rows if int(row["received"]) <= asof_day]


def _to_observation(row: dict[str, str]) -> Observation:
    seen_on = parse_date(row["seen_on"])
    assert seen_on is not None
    first_failed = row["first_failed"]
    return Observation(
        specimen=row["specimen"],
        family=row["family"],
        grade=row["grade"],
        condition=row["condition"],
        entry_age=int(row["entry_age"]),
        last_ok=int(row["last_ok"]),
        first_failed=int(first_failed) if first_failed else None,
        last_seen=int(row["last_seen"]),
        end_reason=row["end_reason"],
        seen_on=seen_on,
        sample_prob=float(row["sample_prob"]),
        seen_day=calendar_to_day(seen_on),
    )


def load_bundle(directory: str | Path) -> DataBundle:
    """复用交换约定的表头/键/引用检查，再包装成 DataBundle。"""
    # 延迟导入，避免包初始化阶段依赖脚本目录。
    from check_sources import load_sources

    base = Path(directory)
    raw = load_sources(base)
    _validate_semantics(raw)
    return DataBundle(base=base, raw=raw)


def _validate_semantics(raw: dict[str, list[dict[str, str]]]) -> None:
    """结构检查之外的业务自洽性检查（不做任何挑批结论）。"""
    for row in raw["observations"]:
        entry_age = int(row["entry_age"])
        last_ok = int(row["last_ok"])
        last_seen = int(row["last_seen"])
        if not entry_age <= last_ok <= last_seen:
            raise ValueError(f"观察 {row['specimen']} 年龄次序不成立")
        if row["first_failed"]:
            first_failed = int(row["first_failed"])
            if not last_ok < first_failed <= last_seen:
                raise ValueError(f"观察 {row['specimen']} 的失效区间不成立")
        if row["end_reason"] not in CENSOR_REASONS and row["end_reason"] != FAILURE:
            raise ValueError(f"观察 {row['specimen']} 的 end_reason 未登记: {row['end_reason']}")
        if not 0.0 < float(row["sample_prob"]) <= 1.0:
            raise ValueError(f"观察 {row['specimen']} 的 sample_prob 越界")
    for row in raw["lab_options"]:
        if int(row["consume"]) <= 0 or int(row["tested_age"]) < 0:
            raise ValueError(f"复查 {row['id']} 的抽离件数/检查龄期不合法")
    for row in raw["supplemental"]:
        if int(row["failed"]) > int(row["checked"]):
            raise ValueError(f"回执 {row['test']} 失效数超过检查数")
    for row in raw["historical_orders"]:
        if int(row["failed"]) > int(row["consumed"]):
            raise ValueError(f"历史订单 {row['id']} 失效数超过已消耗数")
        if int(row["consumed"]) > int(row["qty"]):
            raise ValueError(f"历史订单 {row['id']} 已消耗超过发运数")


def group_by(records: Iterable[Any], key) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = {}
    for record in records:
        out.setdefault(key(record), []).append(record)
    return out
