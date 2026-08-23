r"""协整检验（p3a）：Engle-Granger 双向 + ADF + 半衰期 + 样本内外分离。

口径 v2（预注册，见 docs/COINTEGRATION.md，v1 静态 OOS 已证伪 0/16）:
    - 价格: 对齐后 qfq 收盘 → log 价格
    - EG 双向: log(A)=α+β·log(B)+ε 与反向，取每对最小 ADF p
    - ADF: 残差含常数无趋势，AIC 选滞后（statsmodels adfuller）
    - 多重检验: m = 2×候选对数；Bonferroni α_B=0.05/m；另报 BH-FDR
    - IS(2017-2021) 筛选: p<α_B 且 1≤hl≤90
    - OOS(2022-2026) 确认 v2: **OOS 重估 (α,β)** → OOS 残差 ADF（FDR）且 hl≤120；
      同时保留 v1 静态 β 结果对照（诚实披露）

输出:
    data/pairs/coint_results.csv   全候选对结果（IS + OOS v1/v2 统计量）
    data/pairs/coint_selected.csv  IS+OOS v2 双通过的稳定对
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ALIGNED = os.path.join(PROJ2, "data", "aligned")
PAIRS = os.path.join(PROJ2, "data", "pairs")

IS_END = "2021-12-31"
HL_MIN, HL_MAX = 1.0, 90.0        # IS 半衰期可行区间
OOS_HL_MAX = 120.0                # OOS 放宽（β 漂移容忍）


def eg_fit(la: np.ndarray, lb: np.ndarray) -> tuple:
    """回归 la = α + β·lb + ε，返回 (beta, alpha, resid)。"""
    X = np.column_stack([np.ones(len(lb)), lb])
    b, *_ = np.linalg.lstsq(X, la, rcond=None)
    return b[1], b[0], la - X @ b


def eg_min_p(la: np.ndarray, lb: np.ndarray) -> tuple:
    """双向 EG，返回 (min_p, beta1, alpha1, hl1)。hl 用方向1价差算。"""
    beta1, alpha1, resid1 = eg_fit(la, lb)
    p1 = adfuller(resid1, autolag="AIC")[1]
    _, _, resid2 = eg_fit(lb, la)
    p2 = adfuller(resid2, autolag="AIC")[1]
    hl = half_life(resid1)
    return min(p1, p2), beta1, alpha1, hl


def half_life(resid: np.ndarray) -> float:
    """AR(1) 半衰期（交易日）。"""
    rho = np.dot(resid[:-1], resid[1:]) / np.dot(resid[:-1], resid[:-1]) if np.dot(resid[:-1], resid[:-1]) > 0 else np.nan
    if abs(rho) <= 0 or abs(rho) == 1:
        return np.nan
    return -np.log(2.0) / np.log(abs(rho))


def process_pair(fn: str) -> dict | None:
    m = pd.read_parquet(os.path.join(ALIGNED, fn))
    m = m[m["date"] <= "2026-08-21"]
    m = m.dropna(subset=["close_a", "close_b"])
    if len(m) < 1500:
        return None
    la = np.log(m["close_a"].to_numpy(dtype=float))
    lb = np.log(m["close_b"].to_numpy(dtype=float))
    is_mask = m["date"].to_numpy() <= np.datetime64(IS_END)

    la_is, lb_is = la[is_mask], lb[is_mask]
    la_oos, lb_oos = la[~is_mask], lb[~is_mask]
    if len(la_is) < 500 or len(la_oos) < 300:
        return None

    # IS 筛选
    min_p, beta1, alpha1, hl_is = eg_min_p(la_is, lb_is)

    # OOS v1: 静态 IS β
    s_oos_static = la_oos - (beta1 * lb_oos + alpha1)
    p_oos_v1 = adfuller(s_oos_static, autolag="AIC")[1]

    # OOS v2: 重估 β（与卡尔曼时变机制一致）
    b_oos, a_oos, resid_oos = eg_fit(la_oos, lb_oos)
    p_oos_v2 = adfuller(resid_oos, autolag="AIC")[1]
    hl_oos = half_life(resid_oos)
    beta_drift = abs(b_oos - beta1) / max(abs(beta1), 1e-9)

    return {
        "pair": fn[:-8], "n_is": len(la_is), "n_oos": len(la_oos),
        "min_p_is": min_p, "beta_is": beta1, "alpha_is": alpha1,
        "hl_is": hl_is,
        "p_oos_static": p_oos_v1, "p_oos_rees": p_oos_v2,
        "beta_oos": b_oos, "beta_drift": beta_drift,
        "hl_oos": hl_oos,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--fdr-q", type=float, default=0.05)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(ALIGNED) if f.endswith(".parquet"))
    if args.max_pairs:
        files = files[:args.max_pairs]
    print(f"[pairs] 待检验 {len(files)} 对")

    t0 = time.time()
    results = []
    for i, fn in enumerate(files, 1):
        try:
            r = process_pair(fn)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  [ERR] {fn}: {type(e).__name__} {str(e)[:80]}")
        if i % 500 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] elapsed={(time.time()-t0)/60:.1f}min")

    df = pd.DataFrame(results)
    m = 2 * len(df)                       # 双向 → m 翻倍
    alpha_b = 0.05 / m if m > 0 else np.nan
    df["pass_bonf_is"] = df["min_p_is"] < alpha_b
    df["pass_hl_is"] = (df["hl_is"] >= HL_MIN) & (df["hl_is"] <= HL_MAX)
    df["pass_is"] = df["pass_bonf_is"] & df["pass_hl_is"]

    # OOS v2 FDR（只对 IS 通过的对子做确认）
    sel = df[df["pass_is"]].copy().sort_values("p_oos_rees")
    n_sel = len(sel)
    if n_sel:
        sel["fdr_thresh_oos"] = np.arange(1, n_sel + 1) / n_sel * args.fdr_q
        sel["pass_oos"] = (sel["p_oos_rees"] < sel["fdr_thresh_oos"]) \
            & (sel["hl_oos"] >= 0) & (sel["hl_oos"] <= OOS_HL_MAX)
    else:
        sel["fdr_thresh_oos"] = np.nan
        sel["pass_oos"] = False

    df = df.merge(sel[["pair", "fdr_thresh_oos", "pass_oos"]], on="pair", how="left")
    df["pass_oos"] = df["pass_oos"].fillna(False)

    out = os.path.join(PAIRS, "coint_results.csv")
    df.sort_values("min_p_is").to_csv(out, index=False, encoding="utf-8-sig")
    final = df[df["pass_oos"]].sort_values("p_oos_rees")
    final.to_csv(os.path.join(PAIRS, "coint_selected.csv"), index=False, encoding="utf-8-sig")

    print(f"\n[m] 测试数={m}  Bonferroni α_B={alpha_b:.2e}")
    print(f"[IS] 通过 Bonf+hl: {df['pass_is'].sum()}/{len(df)}")
    print(f"[OOS v2] 双通过（重估 β，稳定协整对）: {len(final)}")
    print(f"[对照] v1 静态 OOS p<0.05: {(df['p_oos_static'] < 0.05).sum()}")
    if not final.empty:
        print(final[["pair", "min_p_is", "hl_is", "p_oos_rees", "hl_oos",
                     "beta_is", "beta_oos", "beta_drift"]].head(20).to_string(index=False))
    print(f"DONE -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

