r"""快速检查回测指标（v2 修复后）。"""
import pandas as pd

import sys

for tag in (sys.argv[1:] or ["v3"]):
    m = pd.read_csv(rf"project2_pairs\reports\pair_metrics_{tag}.csv",
                    index_col=0, header=None).iloc[:, 0]
    m = m.drop(index=["monthly", "params"], errors="ignore").astype(float)
    print(f"== {tag} ==")
    print(f"  avg_turnover: {m['avg_turnover']:.4f}")
    print(f"  short_fee_total: {m['short_fee_total']:.4f}")
    print(f"  n_trades: {int(m['n_trades'])}")
    print(f"  total_ret: {m['total_ret']:.2%}  sharpe: {m['sharpe']:.2f}")
    nav = pd.read_csv(rf"project2_pairs\reports\pair_nav_{tag}.csv", index_col=0, parse_dates=True)
    nav.columns = ["nav"]
    nav["year"] = nav.index.year
    print("  年度:")
    for y, g in nav.groupby("year"):
        print(f"    {y}: {g['nav'].iloc[-1]/g['nav'].iloc[0]-1:+.2%}")
    print()
