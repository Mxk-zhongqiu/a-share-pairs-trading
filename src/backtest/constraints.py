"""可交易性约束：涨跌停 / 停牌。

口径（v1，披露局限）:
    - 涨跌停用当日收盘涨跌幅阈值标记（近似全天状态，复牌首日/新股无限制会误标）
    - 停牌 = 当日无行情（close 为 NaN）
    - buyable  = 有行情 且 非涨停
    - sellable = 有行情 且 非跌停
"""
from typing import Dict

import pandas as pd


def tradability_at(close_row: pd.Series,
                   limit_up_row: pd.Series,
                   limit_down_row: pd.Series) -> Dict[str, Dict[str, bool]]:
    """给定某日各股票的收盘价/涨跌停标记，返回 {code: {buyable, sellable}}。

    参数为 pandas Series（index=code）。close 为 NaN 表示停牌/未上市。
    """
    out = {}
    for code in close_row.index:
        has_market = pd.notna(close_row.get(code))
        if not has_market:
            out[code] = {"buyable": False, "sellable": False}
            continue
        up = bool(limit_up_row.get(code, False))
        down = bool(limit_down_row.get(code, False))
        out[code] = {"buyable": not up, "sellable": not down}
    return out
