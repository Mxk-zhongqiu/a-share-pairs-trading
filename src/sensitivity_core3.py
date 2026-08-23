r"""3 对方向 1 核心的参数小扫描（z_entry × z_exit × z_stop × w0）。"""
import itertools
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

GRID = dict(z_entry=[1.5, 2.0, 2.5], z_exit=[0.0, 0.5], z_stop=[3.0, 3.5, 4.0],
            w0=[0.3, 0.5])


def main() -> None:
    close_df = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))
    up = pd.read_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down = pd.read_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    signals = {pr: pd.read_parquet(os.path.join(SIGNALS, f"kalman_{pr}.parquet"))
               for pr in CORE3}

    rows = []
    for combo in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, combo))
        try:
            targets, trades = build_targets(signals, **kw)
            eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=0.08)
            nav, turn, costs, sfee = eng.run(targets)
            m = compute_metrics(nav[nav.index >= pd.Timestamp("2022-01-01")], turn)
            rows.append({**kw, "net_ret": m.get("total_ret", float("nan")),
                         "annual": m.get("annual_ret", float("nan")),
                         "sharpe": m.get("sharpe", float("nan")),
                         "dd": m.get("max_drawdown", float("nan")),
                         "n_trades": len(trades), "short_fee": sfee})
        except Exception as e:
            print(f"[ERR] {kw}: {str(e)[:80]}")

    df = pd.DataFrame(rows).sort_values("net_ret", ascending=False)
    out = os.path.join(REPORTS, "sensitivity_core3.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"DONE -> {out}")
    print(df.head(8).to_string(index=False))
    print(f"正收益格子: {(df['net_ret'] > 0).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
