r"""生成方向 2 对的交换角色卡尔曼信号（协议 v3：按 min-p 方向交易）。

601128_601288: min-p 方向 = 方向2（log(农行) ~ log(常熟)），交换角色后
交易对 = (601288, 601128)，la=log(601288), lb=log(601128)。
同时验证 dir2 p < dir1 p（确认是显著方向），并输出 dir2 的 IS β/α。
"""
import os
import sys

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJ2)
from kalman import kalman_filter, rolling_z, select_lambda, Z_WINDOW  # noqa: E402

ALIGNED = os.path.join(PROJ2, "data", "aligned")
SIGNALS = os.path.join(PROJ2, "data", "signals")
IS_END = "2021-12-31"

PAIR = "601128_601288"        # code_a=常熟银行, code_b=农业银行
SWAP = "601288_601128"        # 交换后: code_a=农业银行, code_b=常熟银行


def main() -> None:
    m = pd.read_parquet(os.path.join(ALIGNED, f"{PAIR}.parquet"))
    m = m.dropna(subset=["close_a", "close_b"])
    dts = pd.to_datetime(m["date"]).to_numpy()
    is_mask = dts <= np.datetime64(IS_END)

    # 方向1（原）：la(常熟) on lb(农行)
    la1 = np.log(m["close_a"].to_numpy(float))
    lb1 = np.log(m["close_b"].to_numpy(float))
    X1 = np.column_stack([np.ones(len(lb1)), lb1])
    b1, *_ = np.linalg.lstsq(X1[is_mask], la1[is_mask], rcond=None)
    from statsmodels.tsa.stattools import adfuller
    p1 = adfuller(la1[is_mask] - X1[is_mask] @ b1, autolag="AIC")[1]

    # 方向2（交换后）：la'(农行) on lb'(常熟)
    la2 = np.log(m["close_b"].to_numpy(float))     # 农行
    lb2 = np.log(m["close_a"].to_numpy(float))     # 常熟
    X2 = np.column_stack([np.ones(len(lb2)), lb2])
    b2, *_ = np.linalg.lstsq(X2[is_mask], la2[is_mask], rcond=None)
    p2 = adfuller(la2[is_mask] - X2[is_mask] @ b2, autolag="AIC")[1]
    print(f"[方向诊断] {PAIR}: dir1(常熟~农行) p={p1:.3e} | dir2(农行~常熟) p={p2:.3e}")
    if p2 > p1:
        print("!! 方向2 不是显著方向，中止"); sys.exit(1)

    # 方向2 卡尔曼（IS 标定 + 前向滤波，与 kalman.py 同口径）
    beta, alpha = b2[1], b2[0]
    R = float(np.var(la2[is_mask] - (beta * lb2[is_mask] + alpha), ddof=1))
    lam, mse = select_lambda(la2[is_mask], lb2[is_mask], R, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    out = kalman_filter(la2, lb2, R, R * lam * np.eye(2),
                        np.array([alpha, beta]), 10.0 * R * np.eye(2))
    z60 = rolling_z(out["resid"], Z_WINDOW)
    df = pd.DataFrame({
        "date": pd.to_datetime(dts), "beta": out["beta"], "alpha": out["alpha"],
        "spread_e": out["resid"], "z60": z60, "la": la2, "lb": lb2,
    })
    os.makedirs(SIGNALS, exist_ok=True)
    out_path = os.path.join(SIGNALS, f"kalman_{SWAP}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[信号] {SWAP} 生成: R={R:.2e} λ={lam} beta0={beta:.3f} alpha0={alpha:.3f}")
    print(f"DONE -> {out_path}")


if __name__ == "__main__":
    main()
