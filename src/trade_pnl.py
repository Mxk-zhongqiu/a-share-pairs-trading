r"""交易级盈亏分析：open→close 配对，按信号价差变动估算每笔 P&L。

验证两个假设:
    A) revert 平仓是否整体盈利（信号均值回归是否有效）
    B) stop 平仓是否吃掉利润（止损/重进机制是否是主要亏损源）
"""
import os

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SIGNALS = os.path.join(PROJ2, "data", "signals")
tr = pd.read_csv(os.path.join(PROJ2, "reports", "pair_trades_default.csv"), encoding="utf-8-sig")
tr["date"] = pd.to_datetime(tr["date"])


def load_spread(pair: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(SIGNALS, f"kalman_{pair}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["spread_e"]]


def main() -> None:
    rows = []
    for pair, g in tr.groupby("pair"):
        s = load_spread(pair)
        open_ev = []
        for _, e in g.iterrows():
            if e["action"] == "open":
                open_ev.append({"date": e["date"], "z": e["z"], "beta": e["beta"],
                                "s": float(s.loc[e["date"], "spread_e"])})
            elif e["action"] in ("close", "close_break") and open_ev:
                o = open_ev.pop(0)
                s_close = float(s.loc[e["date"], "spread_e"]) if e["date"] in s.index else np.nan
                # 方向: open z>0 → 空A多B → 盈利当价差下降（s_close < s_open）
                sign = -1.0 if o["z"] > 0 else 1.0
                pnl_spread = sign * (s_close - o["s"]) if not np.isnan(s_close) else np.nan
                rows.append({"pair": pair, "open": o["date"], "close": e["date"],
                             "z_open": o["z"], "z_close": e["z"], "reason": e["reason"],
                             "pnl_spread": pnl_spread,
                             "days": (e["date"] - o["date"]).days})
    df = pd.DataFrame(rows)
    print(f"配对成功 {len(df)} 笔（open→close）")
    print()
    print("== 按平仓原因 ==")
    for reason, g in df.groupby("reason"):
        p = g["pnl_spread"].dropna()
        print(f"  {reason:<10} n={len(g):>3} 平均价差盈亏={p.mean():+.5f} "
              f"正收益占比={(p > 0).mean():.1%} 合计={p.sum():+.4f}")
    print()
    print("== 全部 ==")
    p_all = df["pnl_spread"].dropna()
    print(f"  n={len(p_all)} 平均={p_all.mean():+.5f} 正收益占比={(p_all>0).mean():.1%}")
    print()
    print("== 持仓天数分布 ==")
    print(df["days"].describe().to_string())
    print()
    print("== 最差 10 笔 ==")
    print(df.nsmallest(10, "pnl_spread")[["pair", "open", "close", "z_open", "z_close", "reason", "pnl_spread"]].to_string(index=False))


if __name__ == "__main__":
    main()
