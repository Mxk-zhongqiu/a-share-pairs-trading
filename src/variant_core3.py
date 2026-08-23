r"""方向 1 核心变体：只用方向 1 IS-FDR 显著的对（600369_601881/600098_600863/600369_601555）。

动机（独立审查驱动）：9 对中有 6 对仅方向 2 显著，而交易方向是方向 1；
方向 1 显著的 3 对在 9 对书中全部盈利。本变体验证"方向 1 质量过滤"假说。
"""
import os
import sys

import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJ2)
from backtest.costs import CostConfig
from backtest.engine_pair import PairBookEngine
from backtest.metrics import compute_metrics
from strategy.pair_strategy import build_targets

SIGNALS = os.path.join(PROJ2, "data", "signals")
BT = os.path.join(PROJ2, "data", "backtest")
REPORTS = os.path.join(PROJ2, "reports")

CORE3 = ["600369_601881", "600098_600863", "600369_601555"]


def main() -> None:
    close_df = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))
    up = pd.read_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down = pd.read_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    signals = {pr: pd.read_parquet(os.path.join(SIGNALS, f"kalman_{pr}.parquet"))
               for pr in CORE3}

    for w0, max_active in [(0.3, 3), (0.4, 3), (0.5, 2), (0.3, 2)]:
        targets, trades = build_targets(signals, w0=w0, max_active=max_active)
        eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=0.08)
        nav, turn, costs, sfee = eng.run(targets)
        m = compute_metrics(nav[nav.index >= pd.Timestamp("2022-01-01")], turn)
        print(f"w0={w0}/a={max_active}: 净值={nav.iloc[-1]:.4f} 年化={m.get('annual_ret'):+.2%} "
              f"夏普={m.get('sharpe'):+.2f} 回撤={m.get('max_drawdown'):.1%} "
              f"交易={len(trades)} 融券费={sfee:.3f}")


if __name__ == "__main__":
    main()
