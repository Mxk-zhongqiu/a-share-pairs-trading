r"""卡尔曼信号分析（p3c）：z-score 分布、开平仓候选次数、β 路径稳定性。

用途: 回测前确认信号可用（无 NaN 断层、z 分布合理、触发次数充足），
     并预扫候选参数组合的"信号级"交易次数（不含成本，快速过滤参数网格）。
"""
import os

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SIGNALS = os.path.join(PROJ2, "data", "signals")
PAIRS = os.path.join(PROJ2, "data", "pairs")
TRADE_START = "2022-01-01"


def count_trades(z: np.ndarray, entry: float, exit_: float,
                 stop: float, max_hold: int) -> int:
    """信号级回合数（简化：z 穿 entry 开仓，回归 exit 或触 stop 或超持有平仓）。"""
    n = len(z)
    trades = 0
    pos = 0            # +1 多 A 空 B, -1 空 A 多 B, 0 空仓
    hold = 0
    for t in range(n):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if z[t] >= entry:
                pos, hold = 1, 0
            elif z[t] <= -entry:
                pos, hold = -1, 0
        else:
            hold += 1
            exit_now = (abs(z[t]) <= exit_) or (abs(z[t]) >= stop) or (hold >= max_hold)
            if exit_now:
                pos, hold = 0, 0
                trades += 1
    if pos != 0:
        trades += 1
    return trades


def main() -> None:
    sel = pd.read_csv(os.path.join(PAIRS, "coint_selected.csv"), encoding="utf-8-sig")
    print(f"{'pair':<14} {'z中位数':>7} {'z|>2占比':>8} {'β漂移幅度':>9} {'β范围':>12}")
    grids = [(1.5, 0.0, 3.0, 60), (2.0, 0.0, 3.5, 60), (2.0, 0.5, 3.5, 60), (2.5, 0.0, 4.0, 60)]
    headers = "  ".join([f"e{en}/x{ex}/s{st}" for en, ex, st, _ in grids])
    print(f"{'pair':<14} {headers}")

    for _, r in sel.iterrows():
        pr = r["pair"]
        df = pd.read_parquet(os.path.join(SIGNALS, f"kalman_{pr}.parquet"))
        df = df[pd.to_datetime(df["date"]) >= TRADE_START]
        z = df["z60"].to_numpy(float)
        beta = df["beta"].to_numpy(float)
        med_z = float(np.nanmedian(np.abs(z)))
        p_gt2 = float(np.nanmean(np.abs(z) > 2.0)) if len(z) else np.nan
        drift = float(np.nanmax(beta) - np.nanmin(beta))
        counts = [count_trades(z, en, ex, st, mh) for en, ex, st, mh in grids]
        print(f"{pr:<14} {med_z:>7.2f} {p_gt2:>7.1%} {drift:>9.3f} "
              f"[{np.nanmin(beta):.2f},{np.nanmax(beta):.2f}]   "
              + "  ".join(f"{c:>4d}" for c in counts))


if __name__ == "__main__":
    main()
