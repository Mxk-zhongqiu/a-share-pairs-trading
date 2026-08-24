r"""OOS 稳健性：选参段 36 格参数逐一在 2024-2026 纯样本外评估（冷启动）。

展示"选参段最优区在 OOS 是否复现"的完整分布（诚实披露：36 格含多重检验，正格需打折）。
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
BT = os.path.join(PROJ2, "data", "backtest_oos")
REPORTS = os.path.join(PROJ2, "reports")
OOS_START = "2024-01-01"


def main() -> None:
    pairs = pd.read_csv(os.path.join(BT, "pair_list.csv"), encoding="utf-8-sig")
    close_df = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))
    up = pd.read_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down = pd.read_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    signals = {}
    for pr in pairs["pair"]:
        p = os.path.join(SIGNALS, f"kalman_{pr}.parquet")
        if os.path.exists(p):
            signals[pr] = pd.read_parquet(p)

    sel = pd.read_csv(os.path.join(REPORTS, "sensitivity_sel2023.csv"), encoding="utf-8-sig")
    rows = []
    for _, r in sel.iterrows():
        kw = dict(z_entry=r["z_entry"], z_exit=r["z_exit"], z_stop=r["z_stop"],
                  w0=r["w0"], max_active=3, trade_start=OOS_START)
        targets, trades = build_targets(signals, **kw)
        eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=0.08)
        nav, turn, costs, sfee = eng.run(targets)
        m = compute_metrics(nav[nav.index >= OOS_START], turn)
        rows.append({"z_entry": r["z_entry"], "z_exit": r["z_exit"],
                     "z_stop": r["z_stop"], "w0": r["w0"],
                     "sel_sharpe": r["sharpe"],
                     "oos_net": m.get("total_ret", float("nan")),
                     "oos_sharpe": m.get("sharpe", float("nan")),
                     "oos_dd": m.get("max_drawdown", float("nan")),
                     "n_trades": len(trades)})

    df = pd.DataFrame(rows)
    out = os.path.join(REPORTS, "oos_robustness.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"DONE -> {out}")
    print(f"OOS 正收益格子: {(df['oos_sharpe'] > 0).sum()}/{len(df)}")
    print(f"OOS 夏普分布: 中位数 {df['oos_sharpe'].median():+.2f} | "
          f"25分位 {df['oos_sharpe'].quantile(0.25):+.2f} | 75分位 {df['oos_sharpe'].quantile(0.75):+.2f}")
    print("\n== 选参段 top10 的 OOS 表现 ==")
    print(df.sort_values("sel_sharpe", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
