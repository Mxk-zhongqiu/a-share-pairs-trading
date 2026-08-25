r"""动态配对回测（P4）：每期只交易重选入选对，2024-2026 冷启动纯样本外。

信号: 重选窗口静态 β 价差 s = la − β̂·lb − α̂，z = 滚动 60 日（只用 t 及以前）。
交易: 冻结参数 z_entry/z_exit/z_stop/w0/max_active（与静态书一致）。
风控: 回归平仓 / z 止损+冷却 / 持有上限（破裂检验省略——动态重选本身即"关系失效即换"）。
"""
import os
import sys

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJ2)
from backtest.costs import CostConfig
from backtest.engine_pair import PairBookEngine
from backtest.metrics import compute_metrics

PAIRS_DIR = os.path.join(PROJ2, "data", "pairs")
REPORTS = os.path.join(PROJ2, "reports")
OUT_DIR = os.path.join(PROJ2, "data", "backtest_dyn")
Z_WIN = 60
OOS_START = "2024-01-01"
BETA_CLIP = (0.25, 2.5)


def find_qfq(code: str) -> str | None:
    roots = (os.path.join(PROJ2, "data", "raw_fixed", "qfq"),
             r"F:\Deepseekwork\秋招\project1_factor\data\raw\qfq",
             os.path.join(PROJ2, "data", "raw_ext", "qfq"))
    for root in roots:
        p = os.path.join(root, f"{code}.parquet")
        if os.path.exists(p):
            return p
    return None


def limit_pct(code: str, d: pd.Timestamp) -> float:
    if code.startswith("688"):
        return 0.20
    if code.startswith(("300", "301")):
        return 0.10 if d < pd.Timestamp("2020-08-24") else 0.20
    return 0.10


def rolling_z(s: np.ndarray, win: int) -> np.ndarray:
    z = np.full(len(s), np.nan)
    for t in range(win, len(s)):
        w = s[t - win:t]
        sd = w.std(ddof=0)
        z[t] = (s[t] - w.mean()) / sd if sd > 1e-12 else np.nan
    return z


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sel = pd.read_csv(os.path.join(PAIRS_DIR, "dynamic_selections.csv"), encoding="utf-8-sig")
    sel["re_date"] = pd.to_datetime(sel["re_date"])
    periods = sorted(sel["re_date"].unique())
    print(f"[periods] {len(periods)} 期重选: {[str(p.date()) for p in periods]}")

    # 收集所有涉及股票 → close/limit 矩阵
    all_codes = sorted(set(sel["pair"].str.split("_").str[0]) | set(sel["pair"].str.split("_").str[1]))
    closes, ups, downs = {}, {}, {}
    for c in all_codes:
        p = find_qfq(c)
        if p is None:
            continue
        df = pd.read_parquet(p, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"]).sort_values()
        df = df.sort_values("date").set_index("date")["close"].rename(c)
        ret = df.pct_change()
        up = pd.Series(False, index=df.index)
        dn = pd.Series(False, index=df.index)
        for d, r in ret.items():
            if pd.isna(r):
                continue
            l = limit_pct(c, d)
            if r >= l - 0.002:
                up[d] = True
            elif r <= -(l - 0.002):
                dn[d] = True
        up.name = c; dn.name = c
        closes[c] = df; ups[c] = up; downs[c] = dn
    close_df = pd.concat(closes.values(), axis=1).sort_index()
    up_df = pd.concat(ups.values(), axis=1).reindex(close_df.index).fillna(False).astype(bool)
    dn_df = pd.concat(downs.values(), axis=1).reindex(close_df.index).fillna(False).astype(bool)
    print(f"[matrix] {close_df.shape}")

    # 逐期构建信号与目标
    targets: list = []
    all_dates = close_df.index
    for pi, (re_d, nxt_d) in enumerate(zip(periods, periods[1:] + [None])):
        psel = sel[sel["re_date"] == re_d]
        if nxt_d is not None and nxt_d <= pd.Timestamp(OOS_START):
            continue   # 选参段期（2024 前）不交易，仅预热
        trade_end = nxt_d if nxt_d is not None else all_dates[-1]
        print(f"[period {pi}] {re_d.date()} -> {trade_end.date()}: {len(psel)} 对")

        # 每对在交易期内生成信号（静态 β 价差 + 滚动 z）
        sig = {}
        for _, r in psel.iterrows():
            ca, cb = r["pair"].split("_")
            if ca not in closes or cb not in closes:
                continue
            m = pd.concat([closes[ca], closes[cb]], axis=1).dropna()
            m = m[(m.index > re_d) & (m.index <= trade_end)]
            if len(m) < 100:
                continue
            la = np.log(m.iloc[:, 0].to_numpy(float))
            lb = np.log(m.iloc[:, 1].to_numpy(float))
            s = la - (r["beta"] * lb + r["alpha"])
            z = rolling_z(s, Z_WIN)
            sig[r["pair"]] = pd.DataFrame({"date": m.index, "z": z, "beta": r["beta"]})

        # 状态机（与静态策略一致：entry/exit/stop/hold/cool；平仓日也输出目标）
        pos = {pr: 0 for pr in sig}
        hold = {pr: 0 for pr in sig}
        cool = {pr: 0 for pr in sig}
        entry_beta = {}
        last_tgt = {}
        period_dates = [d for d in close_df.index
                        if d > re_d and d <= trade_end and d >= pd.Timestamp(OOS_START)]
        for d in period_dates:
            active_count = sum(1 for v in pos.values() if v != 0)
            for pr in list(pos):
                if cool[pr] > 0:
                    cool[pr] -= 1
                    continue
                row = sig[pr].loc[d] if d in sig[pr].index else None
                if row is None or pd.isna(row["z"]):
                    continue
                z = float(row["z"])
                if pos[pr] == 0:
                    if active_count >= 3:
                        continue
                    if z >= 2.0:
                        pos[pr], hold[pr] = 1, 1
                        entry_beta[pr] = float(row["beta"])
                    elif z <= -2.0:
                        pos[pr], hold[pr] = -1, 1
                        entry_beta[pr] = float(row["beta"])
                else:
                    hold[pr] += 1
                    reason = None
                    if abs(z) <= 0.5:
                        reason = "revert"
                    elif abs(z) >= 3.0:
                        reason = "stop"
                    elif hold[pr] > 60:
                        reason = "timeout"
                    if reason:
                        pos[pr], hold[pr] = 0, 0
                        if reason == "stop":
                            cool[pr] = 10
                        entry_beta.pop(pr, None)
            tgt = {}
            for pr, pv in pos.items():
                if pv == 0:
                    continue
                b = float(np.clip(abs(entry_beta[pr]), *BETA_CLIP))
                wA = 0.3 / (1 + b); wB = 0.3 * b / (1 + b)
                ca, cb = pr.split("_")
                if pv == 1:
                    tgt[ca] = tgt.get(ca, 0.0) - wA
                    tgt[cb] = tgt.get(cb, 0.0) + wB
                else:
                    tgt[ca] = tgt.get(ca, 0.0) + wA
                    tgt[cb] = tgt.get(cb, 0.0) - wB
            if tgt != last_tgt:
                targets.append((d, tgt))
                last_tgt = tgt

    # 回测
    eng = PairBookEngine(close_df, up_df, dn_df, CostConfig(), short_fee_rate=0.08)
    nav, turn, costs, sfee = eng.run(targets)
    nav_oos = nav[nav.index >= OOS_START]
    m = compute_metrics(nav_oos, turn)
    print(f"\n[动态配对 OOS 2024-2026] 净值={nav_oos.iloc[-1]:.4f} 总收益={m.get('total_ret'):+.2%} "
          f"年化={m.get('annual_ret'):+.2%} 夏普={m.get('sharpe'):+.2f} 回撤={m.get('max_drawdown'):.1%}")
    nav.to_csv(os.path.join(REPORTS, "dyn_nav.csv"))
    print(f"DONE -> reports/dyn_nav.csv (交易目标 {len(targets)} 个调仓日)")


if __name__ == "__main__":
    main()
