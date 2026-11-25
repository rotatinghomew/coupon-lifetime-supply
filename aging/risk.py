"""合同使用窗口上的段失效风险与质保界线。

时间轴纪律：库存 ``age`` 以 ``available_at`` 为基准（项目时钟天），合同保障
时长从抵达日起算。质保（warranty）只改变申诉覆盖，绝不改变物理失效曲线。
"""

from __future__ import annotations

from dataclasses import dataclass

from .survival import (
    CURRENT_POOLED,
    DIRECT,
    FULL_POOLED,
    REFERENCE_POOLED,
    THIN_CURRENT_POOL,
    UNESTIMABLE,
    FitBundle,
)

# 单家族直配的最低证据量；低于则回退同层池化。
MIN_DIRECT_EVENTS = 3
MIN_DIRECT_SPECIMENS = 5

PASS = "pass"                    # 充分证据下，段风险 95% 上界仍在限值内
BORDERLINE = "borderline"        # 点估计达标但不确定性上界超限
REJECT = "reject"                # 点估计已超限
NO_EVIDENCE = "no_evidence"      # 该层/该龄段无法估计
INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # 当前生产线证据太薄，不能放行
REFERENCE_ONLY = "reference_only"  # 只有参考家族/全层证据，不能据此放行

# 只有这些证据层级允许给出 pass。
RELEASE_LEVELS = frozenset({DIRECT, CURRENT_POOLED})


@dataclass(frozen=True)
class Lot:
    id: str
    family: str
    grade: str
    condition: str
    qty: int
    age: int
    site: str
    available_at: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Lot":
        return cls(
            id=row["id"],
            family=row["family"],
            grade=row["grade"],
            condition=row["condition"],
            qty=int(row["qty"]),
            age=int(row["age"]),
            site=row["site"],
            available_at=int(row["available_at"]),
        )


@dataclass(frozen=True)
class ContractWindow:
    """一批库存若发给该合同，其贴片年龄经过的完整保障窗口。"""

    contract_id: str
    depart_day: int
    travel: int
    duration: int
    age_at_depart: int
    age_at_arrival: int
    age_at_end: int

    @property
    def segment(self) -> tuple[int, int]:
        return self.age_at_arrival, self.age_at_end


def lot_age_on_day(lot: Lot, project_day: int) -> int:
    """库存 age 是以 available_at 为基准的天数值，项目时钟逐日累加。"""
    return lot.age + (project_day - lot.available_at)


def window_for(lot: Lot, contract: dict[str, str]) -> ContractWindow:
    depart_day = int(contract["depart_day"])
    travel = int(contract["travel"])
    duration = int(contract["duration"])
    age_at_depart = lot_age_on_day(lot, depart_day)
    return ContractWindow(
        contract_id=contract["id"],
        depart_day=depart_day,
        travel=travel,
        duration=duration,
        age_at_depart=age_at_depart,
        age_at_arrival=age_at_depart + travel,
        age_at_end=age_at_depart + travel + duration,
    )


# ---------------------------------------------------------------------------
# 拟合注册表：同层池化 + 家族直配 + 分层自助法，带缓存供全流程复用
# ---------------------------------------------------------------------------


class FitRegistry:
    """四级证据注册表：同家族直配 → 当前家族同层 → 全层池化 → 仅参考家族。"""

    def __init__(self, bundles: dict[tuple[str, str, str], FitBundle],
                 current_pooled: dict[tuple[str, str], FitBundle],
                 full_pooled: dict[tuple[str, str], FitBundle],
                 reference_pooled: dict[tuple[str, str], FitBundle]):
        self._bundles = bundles
        self._current_pooled = current_pooled
        self._full_pooled = full_pooled
        self._reference_pooled = reference_pooled

    def direct(self, lot: Lot) -> FitBundle | None:
        return self._bundles.get((lot.family, lot.grade, lot.condition))

    def for_lot(self, lot: Lot) -> FitBundle:
        """该批的主模型（用于点估计与 UCB）；证据层级在 evidence_level。"""
        direct = self.direct(lot)
        if (direct is not None
                and direct.fit.n_events >= MIN_DIRECT_EVENTS
                and direct.fit.n_specimens >= MIN_DIRECT_SPECIMENS):
            return direct
        pooled = self._current_pooled.get((lot.grade, lot.condition))
        if pooled is not None:
            return pooled
        full = self._full_pooled.get((lot.grade, lot.condition))
        if full is not None:
            return full
        reference = self._reference_pooled.get((lot.grade, lot.condition))
        if reference is not None:
            return reference
        raise KeyError(f"层 {(lot.grade, lot.condition)} 没有任何证据")

    def all_families(self) -> list[FitBundle]:
        return list(self._bundles.values())

    def evidence_chain(self, lot: Lot) -> list[FitBundle]:
        """完整证据链，供报告展示直配/当前池化/全层/参考各自的结论。"""
        chain = []
        direct = self.direct(lot)
        if direct is not None:
            chain.append(direct)
        for store in (self._current_pooled, self._full_pooled, self._reference_pooled):
            bundle = store.get((lot.grade, lot.condition))
            if bundle is not None:
                chain.append(bundle)
        return chain


# ---------------------------------------------------------------------------
# 段风险与判定
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentRisk:
    lot_id: str
    contract_id: str
    start_age: int
    end_age: int
    point: float                 # 条件段失效风险点估计
    ucb95: float                 # 自助法 95% 单侧上界
    estimability: str
    risk_limit: float
    family: str
    evidence_level: str          # direct / current_pooled / ...
    support_from: int
    support_to: int
    bootstrap_coverage: float    # 有效自助重抽样比例
    direct_n_events: int         # 同家族自身事件数（可能为 0–2，不足以放行）
    verdict_override: str | None = None  # 复查融合后的显式判定覆写
    fused_from: str | None = None        # 融合所用复查动作 id

    @property
    def used_pooled(self) -> bool:
        return self.evidence_level != DIRECT

    @property
    def base_verdict(self) -> str:
        if self.estimability == UNESTIMABLE:
            return NO_EVIDENCE
        if self.evidence_level in (FULL_POOLED, REFERENCE_POOLED):
            return REFERENCE_ONLY
        if self.evidence_level == THIN_CURRENT_POOL:
            # 薄池化的 UCB 通常退到 1：信息不足以支持放行，不能装精确。
            return INSUFFICIENT_EVIDENCE
        if self.point > self.risk_limit:
            return REJECT
        if self.ucb95 > self.risk_limit:
            return BORDERLINE
        return PASS

    @property
    def verdict(self) -> str:
        return self.verdict_override or self.base_verdict


def assess_segment(bundle: FitBundle, lot: Lot, contract: dict[str, str],
                   direct_bundle: FitBundle | None = None) -> SegmentRisk:
    window = window_for(lot, contract)
    start_age, end_age = window.segment
    point = bundle.failure_curve(start_age, end_age)
    ucb = bundle.ucb95(start_age, end_age)
    estimability = bundle.estimability(start_age, end_age)
    n_boot = max(bundle.bootstrap_rates.size, 1)
    coverage = bundle.bootstrap_rates.size / n_boot
    return SegmentRisk(
        lot_id=lot.id,
        contract_id=contract["id"],
        start_age=start_age,
        end_age=end_age,
        point=point,
        ucb95=ucb,
        estimability=estimability,
        risk_limit=float(contract["risk_limit"]),
        family=lot.family,
        evidence_level=bundle.evidence_level,
        support_from=bundle.support_from,
        support_to=bundle.support_to,
        bootstrap_coverage=coverage,
        direct_n_events=direct_bundle.fit.n_events if direct_bundle else 0,
    )


# ---------------------------------------------------------------------------
# 质保界线：只影响申诉覆盖
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarrantyCover:
    warranty_id: str
    claim_until: int               # 保修条款长度（天）
    covered_days: int              # 合同使用段落在申诉覆盖内的天数
    uncovered_days: int            # 物理上仍在保障、但申诉不覆盖的尾部天数
    claim_end_age: int
    note: str


def warranty_cover(contract: dict[str, str], warranty_rows: list[dict[str, str]],
                   window: ContractWindow) -> WarrantyCover:
    term_id = contract["warranty_term"]
    term = next(row for row in warranty_rows if row["id"] == term_id)
    claim_until = int(term["claim_until"])
    duration = int(contract["duration"])
    covered_days = min(claim_until, duration)
    uncovered_days = max(0, duration - claim_until)
    claim_end_age = window.age_at_arrival + claim_until
    if uncovered_days:
        note = (f"保修 {term_id} 仅覆盖抵达后 {claim_until} 天，合同保障 {duration} 天；"
                f"最后 {uncovered_days} 天物理风险仍在但无申诉覆盖")
    else:
        note = f"保修 {term_id} 覆盖整个合同使用段"
    return WarrantyCover(
        warranty_id=term_id,
        claim_until=claim_until,
        covered_days=covered_days,
        uncovered_days=uncovered_days,
        claim_end_age=claim_end_age,
        note=note,
    )


def comparability_note(lot: Lot, bundle: FitBundle,
                       direct: FitBundle | None = None) -> str:
    """使用条件可比性说明：证据条件与库存条件是否一致、支撑范围到哪里。"""
    level_cn = {
        DIRECT: "同家族直配证据",
        CURRENT_POOLED: "当前生产线同层池化证据",
        THIN_CURRENT_POOL: "当前生产线同层薄池化证据（试件不足，不能单独放行）",
        FULL_POOLED: "全层池化证据（含参考生产线，基线不同，仅作参照）",
        REFERENCE_POOLED: "参考家族同层证据（不能单独用于放行）",
    }
    origin = level_cn.get(bundle.evidence_level, bundle.evidence_level)
    note = (f"{origin}（{lot.grade}/{lot.condition}），观察支撑龄期 "
            f"{bundle.support_from}–{bundle.support_to} 天")
    if direct is not None and bundle.evidence_level != DIRECT:
        note += (f"；同家族自身仅 {direct.fit.n_specimens} 试件/"
                 f"{direct.fit.n_events} 失效事件，不足以直配放行")
    return note
