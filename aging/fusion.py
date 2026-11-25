"""实际复查回执与现场生存证据的融合。

融合规则（防止复查证据被过度外推）：

1. 复查是在贴片年龄 ``tested_age`` 时点的二项检验（checked 件中 failed 件），
   用 Jeffreys Beta 后验描述该龄期失效率的不确定性。
2. 主模型（指数生存曲线）给出先验，检查结果按贝叶斯方式更新"截至 tested_age
   的累积失效率"，再反解等效风险率，得到融合后的条件段风险。
3. 融合证据只对末端龄期不超过 tested_age 的合同窗口生效；窗口超出该龄期的
   部分不被这次快速检查覆盖，维持原生存模型判定并保留覆盖提醒。
4. 未观测的复查不产生任何融合（只保留 pass/fail 分支）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist

from .recheck import LabTest, RecheckResult
from .survival import FitBundle


@dataclass(frozen=True)
class FusedEvidence:
    test_id: str
    lot_id: str
    tested_age: int
    checked: int
    failed: int
    cp_upper95: float
    prior_cum_failure: float          # 主模型在 tested_age 的累积失效率
    posterior_cum_failure: float      # Jeffreys 后验均值
    posterior_upper95: float          # 后验 95% 分位
    fused_rate: float                 # 反解的等效指数风险率
    covers_window_end: int            # 该证据能覆盖的最大窗口末端龄期


def jeffreys_posterior(failed: int, checked: int) -> tuple[float, float, float]:
    """Jeffreys Beta(0.5,0.5) 先验下的 (后验均值, 95% 分位, 后验参数 a, b)。"""
    a = failed + 0.5
    b = checked - failed + 0.5
    mean = a / (a + b)
    upper = float(beta_dist.ppf(0.95, a, b))
    return mean, upper, a


def fuse_result(bundle: FitBundle, test: LabTest,
                result: RecheckResult) -> FusedEvidence:
    # 主模型在检查龄期的累积失效率先验均值。
    prior = float(bundle.fit.cumulative_failure(test.tested_age))
    post_mean, post_upper, _ = jeffreys_posterior(result.failed, result.checked)

    # 以检查证据为主（它直接针对该批），与先验做 0.5/0.5 的对数空间收缩，
    # 避免 n=6 的单批结果完全压过 76 试件的层证据，也避免层证据淹没批特异性。
    blended = 0.5 * post_mean + 0.5 * prior
    blended = min(max(blended, 1e-6), 1.0 - 1e-9)
    # 反解等效指数风险率：F(t)=1-exp(-r·t)。
    fused_rate = -math.log(1.0 - blended) / max(test.tested_age, 1)
    return FusedEvidence(
        test_id=test.id,
        lot_id=test.lot,
        tested_age=test.tested_age,
        checked=result.checked,
        failed=result.failed,
        cp_upper95=result.cpb_upper95,
        prior_cum_failure=round(prior, 4),
        posterior_cum_failure=round(post_mean, 4),
        posterior_upper95=round(post_upper, 4),
        fused_rate=fused_rate,
        covers_window_end=test.tested_age,
    )


def fused_segment_risk(fused: FusedEvidence, start_age: int, end_age: int,
                       bootstrap_rates: np.ndarray,
                       rng: np.random.Generator, n_sim: int = 1000) -> tuple[float, float]:
    """融合证据下的条件段风险（点估计, 95% 单侧上界）。

    点估计用等效风险率；不确定性通过对 Jeffreys 后验失效率抽样反解风险率，
    再与主模型自助风险率等权混合得到。
    """
    point = 1.0 - math.exp(-fused.fused_rate * max(0, end_age - start_age))

    # 后验失效率抽样 -> 风险率。
    a = fused.failed + 0.5
    b = fused.checked - fused.failed + 0.5
    post_draws = beta_dist.rvs(a, b, size=n_sim, random_state=rng)
    post_rates = -np.log(np.clip(1.0 - post_draws, 1e-9, 1.0)) / max(fused.tested_age, 1)
    # 主模型自助风险率（若可用），等权混合。
    if bootstrap_rates.size:
        sampled_prior = rng.choice(bootstrap_rates, size=n_sim, replace=True)
        rates = 0.5 * post_rates + 0.5 * sampled_prior
    else:
        rates = post_rates
    risks = 1.0 - np.exp(-rates * max(0, end_age - start_age))
    return float(point), float(np.quantile(risks, 0.95))
