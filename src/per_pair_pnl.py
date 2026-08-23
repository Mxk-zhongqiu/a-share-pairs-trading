"""每对回合 P&L 分解。"""
import pandas as pd

df = pd.read_csv(r"F:\Deepseekwork\秋招\project2_pairs\reports\roundtrip_pnl_v5.csv",
                 encoding="utf-8-sig")
for pair, g in df.groupby("pair"):
    wins = g[g["pnl"] > 0]
    losses = g[g["pnl"] <= 0]
    print(f"{pair}: n={len(g):>3} 总={g['pnl'].sum():+.4f} "
          f"胜率={(g['pnl']>0).mean():.0%} 赢均={wins['pnl'].mean():+.5f} "
          f"输均={losses['pnl'].mean():+.5f} 盈亏比={abs(wins['pnl'].mean()/losses['pnl'].mean()) if len(losses) else float('inf'):.2f}")
