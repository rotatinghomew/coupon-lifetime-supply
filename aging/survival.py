"""带左截断与区间删失的失效时间估计。

业务事实到统计量的映射：

- 失效只知道夹在 ``last_ok`` 与 ``first_failed`` 之间 -> 区间删失（不是恰在
  ``first_failed`` 那一刻失效）。
- 未观察到失效的试件在 ``last_seen`` 右删失；``sold / replaced /
  returned_other / monitoring`` 是不同的观察终止机制，基础分析统一按右删失
  处理并分别计数，``pessimistic`` 敏感性模式把售出/换件/其他退回视作"可能已
  失效"（区间 [last_ok, last_seen] 内）。
- 试件在 ``entry_age`` 才进入观察 -> 左截断，似然必须以"能活到 entry_age"为
  条件，否则晚加入的批号会被误当成从零龄随机抽样。
- ``sample_prob`` 是该来源抽到该片的概率 -> 逆概率加权，并报告有效样本量，
  不把加权后的权重当作真实件数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .data import Observation, PESSIMISTIC_REASONS

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 观察区间构造
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """一个试件对似然贡献的区间与截断条件。"""

    left: int            # 已知正常年龄（区间删失下界）；右删失时等于 right
    right: int | None    # 已知失效年龄上界；None 表示右删失（在 left 仍正常）
    entry: int           # 左截断点
    weight: float        # 逆抽样概率权重（后续再归一化）
    reason: str


def build_intervals(
    observations: list[Observation],
    pessimistic: bool = False,
) -> list[Interval]:
    intervals: list[Interval] = []
    for obs in observations:
        weight = 1.0 / max(obs.sample_prob, _EPS)
        if obs.is_failure:
            intervals.append(
                Interval(obs.last_ok, obs.first_failed, obs.entry_age, weight, obs.end_reason)
            )
        elif pessimistic and obs.end_reason in PESSIMISTIC_REASONS:
            # 悲观敏感性：终止时状态未知，当作 [last_ok, last_seen] 内可能失效。
            intervals.append(
                Interval(obs.last_ok, obs.last_seen, obs.entry_age, weight, obs.end_reason)
            )
        else:
            intervals.append(
                Interval(obs.last_seen, None, obs.entry_age, weight, obs.end_reason)
            )
    return intervals


def normalized_weights(intervals: list[Interval]) -> np.ndarray:
    """逆概率权重归一化到有效样本量 ESS=(Σw)²/Σw²，保留相对信息量。"""
    raw = np.array([item.weight for item in intervals], dtype=float)
    ess = raw.sum() ** 2 / np.square(raw).sum()
    return raw / raw.sum() * ess


# ---------------------------------------------------------------------------
# Weibull 极大似然（shape k, scale λ；S(t)=exp[-(t/λ)^k]）
# ---------------------------------------------------------------------------


def _weibull_survival(t: np.ndarray | float, k: float, lam: float) -> np.ndarray | float:
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    with np.errstate(over="ignore"):
        exponent = np.clip(np.power(t / lam, k), 0.0, 700.0)
    return np.exp(-exponent)


def weibull_neg_loglik(params: np.ndarray, intervals: list[Interval], weights: np.ndarray) -> float:
    log_k, log_lam = params
    # 限制参数域，防止优化器走到下溢/上溢区域。
    log_k = float(np.clip(log_k, -7.0, 7.0))
    log_lam = float(np.clip(log_lam, -7.0, 9.0))
    k, lam = math.exp(log_k), math.exp(log_lam)

    def log_surv(t: float) -> float:
        log_t = math.log(max(t, 0.0) + _EPS) - math.log(lam)
        exponent = min(math.exp(min(k * log_t, math.log(700.0))), 700.0)
        return -exponent

    total = 0.0
    for interval, weight in zip(intervals, weights):
        log_at_risk = log_surv(interval.entry)
        if interval.right is None:
            # 右删失：在 left 仍存活，对条件生存 S(left|entry)。
            contribution = log_surv(interval.left) - log_at_risk
        else:
            # log[S(left)-S(right)]；差值因浮点可能为 0，回退到 _EPS 下界。
            log_s_left = log_surv(interval.left)
            log_s_right = log_surv(interval.right)
            inside = 1.0 - math.exp(np.clip(log_s_right - log_s_left, -700.0, 0.0))
            log_mass = log_s_left + math.log(max(inside, _EPS))
            contribution = log_mass - log_at_risk
        total -= weight * contribution
    if not math.isfinite(total):
        return 1e12
    return total


@dataclass
class ExponentialFit:
    """恒定风险（指数）模型：S(t)=exp(-r·t)，小样本区间删失下比自由形状 Weibull 稳。

    回测（31 张冻结时点历史订单）表明：指数 + 非参数自助的 UCB95 经验覆盖率
    约 0.97，而自由形状 Weibull 因 8 试件级别小样本过拟合，点估计 MAE 约为
    指数模型的 1.6 倍。Weibull 保留为形状敏感性对照，不作为放行主模型。
    """

    rate: float
    n_specimens: int
    n_events: int
    ess: float
    converged: bool
    method: str = "exponential"

    @property
    def scale(self) -> float:
        return 1.0 / max(self.rate, _EPS)

    def survival(self, ages: np.ndarray | float) -> np.ndarray | float:
        ages = np.maximum(np.asarray(ages, dtype=float), 0.0)
        return np.exp(-self.rate * ages)

    def cumulative_failure(self, ages: np.ndarray | float) -> np.ndarray | float:
        return 1.0 - self.survival(ages)


def fit_exponential(intervals: list[Interval],
                    weights: np.ndarray | None = None) -> ExponentialFit:
    if not intervals:
        raise ValueError("没有可供拟合的观察记录")
    if weights is None:
        weights = normalized_weights(intervals)
    n_events = sum(item.right is not None for item in intervals)

    def neg_loglik(log_rate: float) -> float:
        rate = math.exp(min(log_rate, math.log(700.0)))

        def log_surv(t: float) -> float:
            return -rate * max(t, 0.0)

        total = 0.0
        for interval, weight in zip(intervals, weights):
            log_at_risk = log_surv(interval.entry)
            if interval.right is None:
                # 条件存活概率 S(left)/S(entry)，对数空间精确。
                contribution = log_surv(interval.left) - log_at_risk
            else:
                log_s_left = log_surv(interval.left)
                log_s_right = log_surv(interval.right)
                inside = 1.0 - math.exp(np.clip(log_s_right - log_s_left, -700.0, 0.0))
                log_diff = log_s_left + math.log(max(inside, _EPS))
                contribution = log_diff - log_at_risk
            total -= weight * contribution
        return total if math.isfinite(total) else 1e12

    # 粗网格定位后在最优格点附近有界精修；网格同时覆盖小风险与大风险区域。
    grid = np.linspace(-8.0, 0.5, 18)
    values = [neg_loglik(float(point)) for point in grid]
    rough = float(grid[int(np.argmin(values))])
    fun, x = minimize_scalar_safe(neg_loglik, rough)
    rate = math.exp(x)
    return ExponentialFit(
        rate=rate,
        n_specimens=len(intervals),
        n_events=n_events,
        ess=float(weights.sum()),
        converged=math.isfinite(fun),
    )


def minimize_scalar_safe(func, start: float) -> tuple[float, float]:
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(func, bounds=(start - 4.0, start + 4.0),
                             method="bounded", options={"xatol": 1e-9})
    return float(result.fun), float(result.x)


@dataclass
class WeibullFit:
    shape: float
    scale: float
    n_specimens: int
    n_events: int
    ess: float
    converged: bool
    method: str = "weibull"

    def survival(self, ages: np.ndarray | float) -> np.ndarray | float:
        return _weibull_survival(ages, self.shape, self.scale)

    def cumulative_failure(self, ages: np.ndarray | float) -> np.ndarray | float:
        return 1.0 - self.survival(ages)


def fit_weibull(intervals: list[Interval], weights: np.ndarray | None = None) -> WeibullFit:
    if not intervals:
        raise ValueError("没有可供拟合的观察记录")
    if weights is None:
        weights = normalized_weights(intervals)
    n_events = sum(item.right is not None for item in intervals)
    ess = float(weights.sum())

    best: tuple[float, np.ndarray] | None = None
    # Weibull 仅作形状敏感性对照，两个起点足够；主模型判定走指数。
    for start_k, start_lam in ((1.0, 45.0), (2.0, 70.0)):
        result = minimize(
            weibull_neg_loglik,
            x0=np.array([math.log(start_k), math.log(start_lam)]),
            args=(intervals, weights),
            method="Nelder-Mead",
            options={"maxiter": 2500, "xatol": 1e-6, "fatol": 1e-6},
        )
        if best is None or result.fun < best[0]:
            best = (float(result.fun), result.x)
    assert best is not None
    log_k, log_lam = best[1]
    return WeibullFit(
        shape=math.exp(log_k),
        scale=math.exp(log_lam),
        n_specimens=len(intervals),
        n_events=n_events,
        ess=ess,
        converged=math.isfinite(best[0]),
    )


# ---------------------------------------------------------------------------
# 加权 Turnbull NPMLE：非参地识别"数据真正支撑到哪个龄期"
# ---------------------------------------------------------------------------


@dataclass
class TurnbullFit:
    grid: np.ndarray            # 年龄网格（含 0）
    mass: np.ndarray            # 各网格点处的失效概率质量
    observed_from: int          # 最小进入观察龄期
    observed_to: int            # 最大确认状态龄期
    n_specimens: int
    n_events: int
    ess: float
    method: str = "turnbull"

    def survival(self, ages: np.ndarray | float) -> np.ndarray | float:
        scalar = np.isscalar(ages)
        age_values = np.atleast_1d(np.asarray(ages, dtype=float))
        # 阶梯生存函数：t 处的 S 是严格大于 t 的网格质量之和。
        out = np.ones_like(age_values)
        for index, age in enumerate(age_values):
            out[index] = float(self.mass[self.grid > age].sum())
        return float(out[0]) if scalar else out

    def cumulative_failure(self, ages: np.ndarray | float) -> np.ndarray | float:
        return 1.0 - self.survival(ages)


def fit_turnbull(intervals: list[Interval], weights: np.ndarray | None = None,
                 max_iter: int = 5000, tol: float = 1e-9) -> TurnbullFit:
    if weights is None:
        weights = normalized_weights(intervals)
    ages = {0}
    for item in intervals:
        ages.add(item.entry)
        ages.add(item.left)
        if item.right is not None:
            ages.add(item.right)
    grid = np.array(sorted(ages), dtype=float)
    mass = np.full_like(grid, 1.0 / len(grid))

    for _ in range(max_iter):
        expected = np.zeros_like(grid)
        for item, weight in zip(intervals, weights):
            if item.right is None:
                # 右删失：质量必须落在 last_seen 之后（含严格大于的网格点）。
                eligible = grid > item.left
                denom = mass[eligible].sum()
                if denom > _EPS:
                    expected[eligible] += weight * mass[eligible] / denom
            else:
                # 区间删失：质量落在 (left, right]。
                eligible = (grid > item.left) & (grid <= item.right)
                denom = mass[eligible].sum()
                if denom > _EPS:
                    expected[eligible] += weight * mass[eligible] / denom
        new_mass = expected / expected.sum()
        if np.abs(new_mass - mass).max() < tol:
            mass = new_mass
            break
        mass = new_mass

    return TurnbullFit(
        grid=grid.astype(int),
        mass=mass,
        observed_from=min(item.entry for item in intervals),
        observed_to=max(max(item.left, item.right or item.left) for item in intervals),
        n_specimens=len(intervals),
        n_events=sum(item.right is not None for item in intervals),
        ess=float(weights.sum()),
    )


# ---------------------------------------------------------------------------
# 拟合束：点估计 + 分层自助法不确定性 + 支撑范围判定
# ---------------------------------------------------------------------------

ESTIMABLE = "estimable"          # 目标段完全落在观察支撑内
EXTRAPOLATED = "extrapolated"    # 部分龄期超出观察支撑，靠参数外推
UNESTIMABLE = "unestimable"      # 该层没有可用证据

# 证据层级（estimator.py 据此组织注册表）。
DIRECT = "direct"
CURRENT_POOLED = "current_pooled"
# 薄池化：当前家族同层试件总量低于该阈值，参数估计不稳定，UCB 通常会退到 1。
THIN_CURRENT_POOL = "thin_current_pooled"
FULL_POOLED = "full_pooled"
REFERENCE_POOLED = "reference_pooled"


@dataclass
class FitBundle:
    """一个（家族 × 层）证据集合的可复用拟合结果。

    主模型是指数（恒定风险），``weibull`` 保留为形状敏感性对照；区间判定与
    UCB 一律走主模型，避免小样本自由形状过拟合。
    """

    family: str
    grade: str
    condition: str
    fit: ExponentialFit
    weibull: WeibullFit
    turnbull: TurnbullFit
    support_from: int
    support_to: int
    reason_counts: dict[str, int]
    sample_probs: list[float]
    bootstrap_rates: np.ndarray = field(default_factory=lambda: np.empty(0))
    bootstrap_shapes: np.ndarray = field(default_factory=lambda: np.empty(0))
    bootstrap_scales: np.ndarray = field(default_factory=lambda: np.empty(0))
    fallback_family: str | None = None
    evidence_level: str = "direct"      # direct / current_pooled / ...

    @property
    def stratum(self) -> tuple[str, str]:
        return (self.grade, self.condition)

    # -- 点估计（主模型：指数）-------------------------------------------------

    def failure_curve(self, start_age: float, end_age: float) -> float:
        """条件段失效风险，指数模型下简化为 1-exp[-r·(end-start)]。"""
        rate = self.fit.rate
        return float(1.0 - math.exp(-rate * max(0.0, end_age - start_age)))

    # -- 自助法（主模型速率）--------------------------------------------------

    def bootstrap_risks(self, start_age: float, end_age: float) -> np.ndarray:
        if self.bootstrap_rates.size == 0:
            return np.empty(0)
        span = max(0.0, end_age - start_age)
        return 1.0 - np.exp(-self.bootstrap_rates * span)

    def ucb95(self, start_age: float, end_age: float) -> float:
        """段风险的自助法 95% 单侧上界；无有效重抽样时返回 1.0（不假装精确）。"""
        risks = self.bootstrap_risks(start_age, end_age)
        if risks.size == 0:
            return 1.0
        return float(np.quantile(risks, 0.95))

    # -- 支撑范围（Turnbull 非参识别）-----------------------------------------

    def estimability(self, start_age: float, end_age: float) -> str:
        if self.fit.n_events == 0:
            return UNESTIMABLE
        if end_age <= self.support_to and start_age >= self.support_from:
            return ESTIMABLE
        if start_age > self.support_to:
            return UNESTIMABLE
        return EXTRAPOLATED


def family_stratified_bootstrap(
    observations: list[Observation],
    n_boot: int,
    rng: np.random.Generator,
    pessimistic: bool = False,
    include_weibull: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """按家族分层有放回重抽样，返回 (指数速率, Weibull shape, Weibull scale, 有效数)。

    主模型是指数，自助路径默认只拟合指数以控制成本；``include_weibull`` 时
    才附带形状敏感性参数。重抽样后整层没有事件的样本跳过，跳过比例由
    FitBundle 的覆盖率体现，不静默填补。
    """
    families: dict[str, list[Observation]] = {}
    for obs in observations:
        families.setdefault(obs.family, []).append(obs)
    family_names = sorted(families)
    rates: list[float] = []
    shapes: list[float] = []
    scales: list[float] = []
    for _ in range(n_boot):
        resampled: list[Observation] = []
        for name in family_names:
            members = families[name]
            picks = rng.integers(0, len(members), size=len(members))
            resampled.extend(members[index] for index in picks)
        intervals = build_intervals(resampled, pessimistic=pessimistic)
        if not any(item.right is not None for item in intervals):
            continue
        weights = normalized_weights(intervals)
        try:
            exp_fit = fit_exponential(intervals, weights)
            if include_weibull:
                wb_fit = fit_weibull(intervals, weights)
        except (ValueError, OverflowError):
            continue
        if not math.isfinite(exp_fit.rate):
            continue
        rates.append(exp_fit.rate)
        if include_weibull and math.isfinite(wb_fit.shape) and math.isfinite(wb_fit.scale):
            shapes.append(wb_fit.shape)
            scales.append(wb_fit.scale)
    return (np.array(rates), np.array(shapes), np.array(scales), len(rates))
