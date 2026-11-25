"""防护贴老化挑批：证据治理、生存估计、段风险与合同分配核对。

模块划分：

- ``data``       资料读入、时点（asof）过滤、同一试件最新行替代、日期/项目时钟换算
- ``survival``   带左截断与区间删失的 Weibull 极大似然、加权 Turnbull、自助法区间
- ``risk``       库存龄期到合同使用段的映射、段失效风险、质保界线与物理风险分离
- ``recheck``    复查抽离、实际回执的二项证据、未观测复查的分支与覆盖龄期
- ``allocation`` 同批上限/不可混装/既有承诺约束、方案分支可行性与重复占用核对
- ``backtest``   只用历史时点已有证据的回看校准，禁止后见之明
- ``report``     汇总人读报告与机读 JSON
"""

from __future__ import annotations

__all__ = [
    "data",
    "survival",
    "risk",
    "recheck",
    "allocation",
    "backtest",
    "report",
]
