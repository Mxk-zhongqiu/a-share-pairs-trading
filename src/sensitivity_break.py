r"""破裂检验参数扫描（补充 P9 网格：报告 §6 局限第 8 条披露的未扫描项）。

网格: break_p {0.01, 0.05, 0.10, 0.20} × break_pause {10, 20, 40}
基座: 终版配置（3 对方向 1 核心, 2.5/0.5/3.5, w0=0.5, max_active=3, 融券费 8%）
"""
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

BASE = dict(z_entry=2.5, z_exit=0.5, z_stop=3.5, max_hold=60, w0=0.5, max_active=3)
GRID = dict(break_p=[0.01, 0.05, 0.10, 0.20], break_pause=[10, 20, 40])


def main() -> None:
    close_df = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))
    up = pd.read_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down = pd.read_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    signals = {pr: pd.read_parquet(os.path.join(SIGNALS, f"kalman_{pr}.parquet"))
               for pr in CORE3}

    rows = []
    for combo in itertools.product(*GRID.values()):
        kw = {**BASE, **dict(zip(GRID, combo))}
        targets, trades = build_targets(signals, **kw)
        eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=0.08)
        nav, turn, costs, sfee = eng.run(targets)
        m = compute_metrics(nav[nav.index >= pd.Timestamp("2022-01-01")], turn)
        n_break = int((trades["action"] == "close_break").sum()) if len(trades) else 0
        rows.append({**dict(zip(GRID, combo)),
                     "net_ret": m.get("total_ret", float("nan")),
                     "annual": m.get("annual_ret", float("nan")),
                     "sharpe": m.get("sharpe", float("nan")),
                     "dd": m.get("max_drawdown", float("nan")),
                     "n_trades": len(trades), "n_break": n_break,
                     "short_fee": sfee})

    df = pd.DataFrame(rows).sort_values("net_ret", ascending=False)
    out = os.path.join(REPORTS, "sensitivity_break.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"DONE -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
