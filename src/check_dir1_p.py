r"""验证 9 个核心对的方向 1（交易方向）IS ADF 是否通过 Bonferroni。"""
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ALIGNED = r"F:\Deepseekwork\秋招\project2_pairs\data\aligned"
IS_END = "2021-12-31"
sel = pd.read_csv(r"F:\Deepseekwork\秋招\project2_pairs\data\pairs\coint_selected.csv",
                  encoding="utf-8-sig")
m_total = 2 * 4862
alpha_b = 0.05 / m_total
print(f"α_B = {alpha_b:.2e}\n")

for _, r in sel.iterrows():
    pr = r["pair"]
    m = pd.read_parquet(os.path.join(ALIGNED, f"{pr}.parquet"))
    m = m.dropna(subset=["close_a", "close_b"])
    la = np.log(m["close_a"].to_numpy(float))
    lb = np.log(m["close_b"].to_numpy(float))
    mask = pd.to_datetime(m["date"]).to_numpy() <= np.datetime64(IS_END)
    la_is, lb_is = la[mask], lb[mask]
    X = np.column_stack([np.ones(len(lb_is)), lb_is])
    b, *_ = np.linalg.lstsq(X, la_is, rcond=None)
    p1 = adfuller(la_is - X @ b, autolag="AIC")[1]
    flag = "PASS" if p1 < alpha_b else "FAIL"
    print(f"{pr}: dir1 IS p={p1:.3e}  {flag}")
