r"""参数敏感性扫描（P9，预注册网格，见 docs/STRATEGY.md §5）。

网格: z_entry {1.5,2.0,2.5} × z_exit {0.0,0.5} × z_stop {3.0,3.5,4.0}
      × max_hold {40,60,90} × w0 {0.2,0.3,0.5}（max_active=3, 融券费 8%）
输出: reports/sensitivity_scan.csv（每格 net_ret/annual/sharpe/dd/turnover/n_trades）
"""
import itertools
import os
import sys
import time

import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJ2)
from backtest.costs import CostConfig                      # noqa: E402
from backtest.engine_pair import PairBookEngine            # noqa: E402
from backtest.metrics import compute_metrics               # noqa: E402
from strategy.pair_strategy import build_targets           # noqa: E402

SIGNALS = os.path.join(PROJ2, "data", "signals")
BT = os.path.join(PROJ2, "data", "backtest")
REPORTS = os.path.join(PROJ2, "reports")

GRID = dict(
    z_entry=[1.5, 2.0, 2.5],
    z_exit=[0.0, 0.5],
    z_stop=[3.0, 3.5, 4.0],
    max_hold=[40, 60, 90],
    w0=[0.2, 0.3, 0.5],
)


def main() -> None:
    close_df = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))
    up = pd.read_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down = pd.read_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    pairs = pd.read_csv(os.path.join(BT, "pair_list.csv"), encoding="utf-8-sig")
    signals = {}
    for pr in pairs["pair"]:
        signals[pr] = pd.read_parquet(os.path.join(SIGNALS, f"kalman_{pr}.parquet"))

    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"[scan] {len(combos)} 组合")

    rows = []
    t0 = time.time()
    for i, combo in enumerate(combos, 1):
        kw = dict(zip(keys, combo))
        try:
            targets, trades = build_targets(signals, **kw)
            eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=0.08)
            nav, turn, costs, sfee = eng.run(targets)
            m = compute_metrics(nav[nav.index >= pd.Timestamp("2022-01-01")], turn)
            rows.append({**kw, "net_ret": m.get("total_ret", float("nan")),
                         "annual": m.get("annual_ret", float("nan")),
                         "sharpe": m.get("sharpe", float("nan")),
                         "dd": m.get("max_drawdown", float("nan")),
                         "turnover": m.get("avg_turnover", float("nan")),
                         "n_trades": len(trades),
                         "short_fee": sfee})
        except Exception as e:
            print(f"  [ERR] {kw}: {type(e).__name__} {str(e)[:80]}")
        if i % 20 == 0 or i == len(combos):
            print(f"[{i}/{len(combos)}] elapsed={(time.time()-t0)/60:.1f}min")

    df = pd.DataFrame(rows)
    out = os.path.join(REPORTS, "sensitivity_scan.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nDONE -> {out} ({len(df)} 行)  ({time.time()-t0:.0f}s)")
    if not df.empty:
        print("\n== 最优 10（按 net_ret）==")
        print(df.sort_values("net_ret", ascending=False).head(10).to_string(index=False))
        print("\n== 最差 5 ==")
        print(df.sort_values("net_ret").head(5).to_string(index=False))
        print("\n== 全网格 net_ret 分布 ==")
        print(df["net_ret"].describe().to_string())
        print(f"\n正收益格子数: {(df['net_ret'] > 0).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
