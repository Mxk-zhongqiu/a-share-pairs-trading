r"""OU 建模（p3b）：对协整对的 IS 价差拟合 Ornstein-Uhlenbeck 参数。

口径（预注册，见 docs/OU_KALMAN.md）:
    价差 s_t = la_t − β̂·lb_t − α̂（β̂,α̂ = IS 全样本 EG 估计）
    离散 AR(1): s_{t+1} = a + b·s_t + ε
    θ = −ln(b);  μ = a/(1−b);  σ_ε = std(ε);  σ_ss = σ_ε/√(1−b²);  hl = −ln2/ln|b|

输出: data/pairs/ou_params.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ALIGNED = os.path.join(PROJ2, "data", "aligned")
PAIRS = os.path.join(PROJ2, "data", "pairs")
IS_END = "2021-12-31"


def fit_ou(s: np.ndarray) -> dict:
    """s: 价差序列（IS）。返回 OU 参数。"""
    s = s[~np.isnan(s)]
    if len(s) < 100:
        return {}
    x, y = s[:-1], s[1:]
    b = np.dot(x, y) / np.dot(x, x) if np.dot(x, x) > 0 else np.nan
    a = np.mean(y) - b * np.mean(x)
    resid = y - (a + b * x)
    sigma_e = float(np.std(resid, ddof=1))
    theta = -np.log(b) if (b > 0 and b != 1) else np.nan
    mu = a / (1 - b) if b != 1 else np.nan
    sigma_ss = sigma_e / np.sqrt(1 - b ** 2) if abs(b) < 1 else np.nan
    hl = -np.log(2.0) / np.log(abs(b)) if (abs(b) > 0 and abs(b) != 1) else np.nan
    return {"b": b, "a": a, "theta": theta, "mu": mu,
            "sigma_e": sigma_e, "sigma_ss": sigma_ss, "hl": hl}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    sel = pd.read_csv(os.path.join(PAIRS, "coint_selected.csv"), encoding="utf-8-sig")
    if args.max_pairs:
        sel = sel.head(args.max_pairs)
    print(f"[pairs] 稳定协整对 {len(sel)}")

    rows = []
    for _, r in sel.iterrows():
        pr = r["pair"]
        m = pd.read_parquet(os.path.join(ALIGNED, f"{pr}.parquet"))
        m = m.dropna(subset=["close_a", "close_b"])
        m = m[m["date"] <= IS_END]
        la = np.log(m["close_a"].to_numpy(float))
        lb = np.log(m["close_b"].to_numpy(float))
        beta, alpha = r["beta_is"], r["alpha_is"]
        s = la - (beta * lb + alpha)
        ou = fit_ou(s)
        if ou:
            rows.append({"pair": pr, **ou})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PAIRS, "ou_params.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"DONE -> data/pairs/ou_params.csv ({len(rows)} 对)")


if __name__ == "__main__":
    main()
