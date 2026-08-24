r"""全量方向 1 IS p 计算（三层协议 tier3 的过滤输入）。

方向 1 = 交易方向（la = α + β·lb，code_a 为被解释变量）。
输出: data/pairs/dir1_p_is.csv（pair, dir1_p_is, dir1_beta, dir1_hl）
"""
import os
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ALIGNED = os.path.join(PROJ2, "data", "aligned")
PAIRS = os.path.join(PROJ2, "data", "pairs")
IS_END = "2021-12-31"


def half_life(resid: np.ndarray) -> float:
    rho = np.dot(resid[:-1], resid[1:]) / np.dot(resid[:-1], resid[:-1]) if np.dot(resid[:-1], resid[:-1]) > 0 else np.nan
    if abs(rho) <= 0 or abs(rho) == 1:
        return np.nan
    return -np.log(2.0) / np.log(abs(rho))


def main() -> None:
    import sys as _s
    aligned = _s.argv[1] if len(_s.argv) > 1 else ALIGNED
    files = sorted(f for f in os.listdir(aligned) if f.endswith(".parquet"))
    rows = []
    t0 = time.time()
    for i, fn in enumerate(files, 1):
        m = pd.read_parquet(os.path.join(aligned, fn))
        m = m[m["date"] <= IS_END].dropna(subset=["close_a", "close_b"])
        if len(m) < 500:
            continue
        la = np.log(m["close_a"].to_numpy(float))
        lb = np.log(m["close_b"].to_numpy(float))
        X = np.column_stack([np.ones(len(lb)), lb])
        b, *_ = np.linalg.lstsq(X, la, rcond=None)
        resid = la - X @ b
        p1 = adfuller(resid, autolag="AIC")[1]
        rows.append({"pair": fn[:-8], "dir1_p_is": p1,
                     "dir1_beta": b[1], "dir1_hl": half_life(resid)})
        if i % 1000 == 0:
            print(f"[{i}/{len(files)}] elapsed={(time.time()-t0)/60:.1f}min")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PAIRS, "dir1_p_is.csv"), index=False, encoding="utf-8-sig")
    print(f"DONE: {len(df)} 对 -> data/pairs/dir1_p_is.csv ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
