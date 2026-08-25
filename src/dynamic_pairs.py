r"""动态配对：每半年滚动重选对子（P4，预注册见 docs/UPGRADE.md）。

协议:
    重选日: 2022-06-30, 2022-12-30, 2023-06-30, 2023-12-29,   (选参段/预热)
            2024-06-28, 2024-12-31, 2025-06-30, 2025-12-31   (验证段，滚动样本外)
    检验:   每个重选日用 [d-250, d] 窗口对 18,734 对跑 EG 双向 + ADF(AIC)
    入选:   min-p < Bonferroni(0.05/(2×N)) 且 5 ≤ hl ≤ 120，按 p 取 top8
    交易:   重选日次日生效，交易参数冻结（策略层处理）
优化: 882 只股票 close 全量载入内存（~17MB），窗口切片做回归。
输出: data/pairs/dynamic_selections.csv（每期入选对 + β/α/hl/p）
"""
import os
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAIRS = os.path.join(PROJ2, "data", "pairs")
ALIGNED = os.path.join(PROJ2, "data", "aligned_ext")
WINDOW = 250
HL_MIN, HL_MAX = 5.0, 120.0
TOP_K = 8

REDATES = ["2022-06-30", "2022-12-30", "2023-06-30", "2023-12-29",
           "2024-06-28", "2024-12-31", "2025-06-30", "2025-12-31"]


def half_life(resid: np.ndarray) -> float:
    rho = np.dot(resid[:-1], resid[1:]) / np.dot(resid[:-1], resid[:-1]) if np.dot(resid[:-1], resid[:-1]) > 0 else np.nan
    if abs(rho) <= 0 or abs(rho) == 1:
        return np.nan
    return -np.log(2.0) / np.log(abs(rho))


def eg_min(la: np.ndarray, lb: np.ndarray) -> tuple:
    """双向 EG: 返回 (min_p, beta1, alpha1, hl1)。"""
    X1 = np.column_stack([np.ones(len(lb)), lb])
    b1, *_ = np.linalg.lstsq(X1, la, rcond=None)
    r1 = la - X1 @ b1
    p1 = adfuller(r1, autolag="AIC")[1]
    X2 = np.column_stack([np.ones(len(la)), la])
    b2, *_ = np.linalg.lstsq(X2, lb, rcond=None)
    r2 = lb - X2 @ b2
    p2 = adfuller(r2, autolag="AIC")[1]
    return min(p1, p2), b1[1], b1[0], half_life(r1)


def find_qfq(code: str) -> str | None:
    roots = (os.path.join(PROJ2, "data", "raw_fixed", "qfq"),
             r"F:\Deepseekwork\秋招\project1_factor\data\raw\qfq",
             os.path.join(PROJ2, "data", "raw_ext", "qfq"))
    for root in roots:
        p = os.path.join(root, f"{code}.parquet")
        if os.path.exists(p):
            return p
    return None


def main() -> None:
    # 1) 全量股票 close 载入内存（多数据根）
    pairs = pd.read_parquet(os.path.join(PAIRS, "candidate_pairs_ext.parquet"))
    codes = sorted(set(pairs["code_a"]) | set(pairs["code_b"]))
    closes = {}
    for c in codes:
        p = find_qfq(c)
        if p is None:
            continue
        df = pd.read_parquet(p, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        closes[c] = df.set_index("date")["close"]
    print(f"[load] {len(closes)}/{len(codes)} 只股票价格")

    # 2) 逐重选日筛选
    rows = []
    t0 = time.time()
    for rdi, rd in enumerate(REDATES):
        d = pd.Timestamp(rd)
        win_start = d - pd.Timedelta(days=400)
        pass_list = []
        for i, (_, pr) in enumerate(pairs.iterrows(), 1):
            ca, cb = pr["code_a"], pr["code_b"]
            if ca not in closes or cb not in closes:
                continue
            la = closes[ca]; lb = closes[cb]
            m = pd.concat([la, lb], axis=1).dropna()
            w = m[(m.index > win_start) & (m.index <= d)]
            if len(w) < 200:
                continue
            la_w = np.log(w.iloc[:, 0].to_numpy(float))
            lb_w = np.log(w.iloc[:, 1].to_numpy(float))
            p, beta, alpha, hl = eg_min(la_w, lb_w)
            if p < 0.05 / (2 * len(pairs)) and HL_MIN <= hl <= HL_MAX:
                pass_list.append((p, pr["code_a"], pr["code_b"], beta, alpha, hl))
            if i % 5000 == 0:
                print(f"  [{rd} {rdi+1}/{len(REDATES)}] {i}/{len(pairs)} pass={len(pass_list)} "
                      f"elapsed={(time.time()-t0)/60:.1f}min")
        pass_list.sort()
        for rank, (p, a, b, beta, alpha, hl) in enumerate(pass_list[:TOP_K], 1):
            rows.append({"re_date": rd, "rank": rank, "pair": f"{a}_{b}",
                         "p": p, "beta": beta, "alpha": alpha, "hl": hl})
        print(f"[{rd}] 通过 {len(pass_list)} 对, 入选 {min(TOP_K, len(pass_list))} | "
              f"elapsed={(time.time()-t0)/60:.1f}min")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PAIRS, "dynamic_selections.csv"), index=False, encoding="utf-8-sig")
    print(f"\nDONE -> data/pairs/dynamic_selections.csv ({len(out)} 行, {(time.time()-t0)/60:.1f}min)")
    print(out.groupby("re_date").size().to_string())


if __name__ == "__main__":
    main()
