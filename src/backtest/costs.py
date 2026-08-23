"""成本模型：佣金 / 印花税（日期分段）/ 滑点（参数化，A股实际口径）。"""
from dataclasses import dataclass

import pandas as pd

STAMP_CUTOFF = pd.Timestamp("2023-08-28")   # 印花税减半生效日


@dataclass
class CostConfig:
    commission_rate: float = 0.00025   # 佣金 万2.5，双边
    stamp_tax: float = 0.0005          # 印花税 卖出（2023-08-28 起 0.05%；此前 0.1%）
    slippage: float = 0.001            # 滑点 单边 0.1%
    date_aware_stamp: bool = True      # 是否按日期分段计印花税（False 时用 stamp_tax 常量）
    # 注: 佣金最低 5 元/笔未建模（组合规模下占比极小），报告披露


def stamp_rate(d) -> float:
    """印花税卖出税率（2023-08-28 前 0.1%，之后 0.05%）。"""
    d = pd.Timestamp(d)
    return 0.0005 if d >= STAMP_CUTOFF else 0.001


def trade_costs(buy_amount: float, sell_amount: float, cfg: CostConfig,
                date=None) -> float:
    """按成交金额计算总成本（占组合市值比例）。date 为空时用 cfg.stamp_tax。"""
    commission = (buy_amount + sell_amount) * cfg.commission_rate
    if date is not None and cfg.date_aware_stamp:
        stamp = sell_amount * stamp_rate(date)
    else:
        stamp = sell_amount * cfg.stamp_tax
    slippage = (buy_amount + sell_amount) * cfg.slippage
    return commission + stamp + slippage
