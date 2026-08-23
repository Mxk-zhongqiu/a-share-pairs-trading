r"""策略级回合 P&L 重建：用 trades 表 + 收盘价计算每笔 open→close 的真实收益。

P&L(trade) = wA·(P_A,c/P_A,o − 1) − wB·(P_B,c/P_B,o − 1)   （NAV 单位, 近似 1）
wA = w0/(1+|β|), wB = w0·|β|/(1+|β|), 方向由 open 的 z 决定。
作用: 判定"信号经济性" vs "引擎执行损耗"哪个是负收益根源。
"""
import os
import sys

import numpy as np
import pandas as pd

REPORTS = r"F:\Deepseekwork\秋招\project2_pairs\reports"
BT = r"F:\Deepseekwork\秋招\project2_pairs\data\backtest"
W0 = 0.3


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v5"
    tr = pd.read_csv(os.path.join(REPORTS, f"pair_trades_{tag}.csv"), encoding="utf-8-sig")
    tr["date"] = pd.to_datetime(tr["date"])
    close = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))

    rows = []
    for pair, g in tr.groupby("pair"):
        ca, cb = pair.split("_")
        open_ev = []
        for _, e in g.iterrows():
            if e["action"] == "open":
                open_ev.append({"date": e["date"], "z": e["z"], "beta": e["beta"]})
            elif e["action"] in ("close", "close_break") and open_ev:
                o = open_ev.pop(0)
                pa0, pa1 = close.loc[o["date"], ca], close.loc[e["date"], ca]
                pb0, pb1 = close.loc[o["date"], cb], close.loc[e["date"], cb]
                if pd.isna(pa0) or pd.isna(pa1) or pd.isna(pb0) or pd.isna(pb1):
                    continue
                b = float(np.clip(abs(o["beta"]), 0.25, 2.5))
                wA, wB = W0 / (1 + b), W0 * b / (1 + b)
                rA, rB = pa1 / pa0 - 1, pb1 / pb0 - 1
                if o["z"] > 0:      # 空 A 多 B
                    pnl = -wA * rA + wB * rB
                else:               # 多 A 空 B
                    pnl = wA * rA - wB * rB
                rows.append({"pair": pair, "open": o["date"], "close": e["date"],
                             "reason": e["reason"], "z_open": o["z"],
                             "beta": o["beta"], "pnl": pnl,
                             "days": (e["date"] - o["date"]).days})

    df = pd.DataFrame(rows)
    print(f"== {tag} ==")
    print(f"回合数: {len(df)}")
    print(f"总 P&L: {df['pnl'].sum():+.4f}  NAV 单位")
    print(f"平均: {df['pnl'].mean():+.4f}  正收益占比: {(df['pnl'] > 0).mean():.1%}")
    print()
    print("== 按原因 ==")
    for reason, g in df.groupby("reason"):
        print(f"  {reason:<10} n={len(g):>3} 总={g['pnl'].sum():+.4f} 平均={g['pnl'].mean():+.5f} 正占比={(g['pnl']>0).mean():.1%}")
    print()
    print("== 按对 ==")
    for pair, g in df.groupby("pair"):
        print(f"  {pair} n={len(g):>3} 总={g['pnl'].sum():+.4f} 平均={g['pnl'].mean():+.5f}")
    print()
    print("== 年度 ==")
    df["year"] = df["open"].dt.year
    for y, g in df.groupby("year"):
        print(f"  {y}: n={len(g):>3} 总={g['pnl'].sum():+.4f}")
    df.to_csv(os.path.join(REPORTS, f"roundtrip_pnl_{tag}.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
