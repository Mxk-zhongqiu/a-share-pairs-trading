r"""协整 OOS 失败机理诊断（16 个 IS 通过对）: β 漂移 vs 关系真破裂。

对每对:
    1) OOS 重估 β（用 OOS 数据回归）→ 与 IS β 对比漂移幅度
    2) OOS 残差（OOS β）ADF → 若显著 => β 漂移假失败（卡尔曼可救）
       若不显著 => 关系真破裂（静态协整不存在，需滚动重选）
    3) IS 与 OOS 的 hl 对比
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ALIGNED = r"F:\Deepseekwork\秋招\project2_pairs\data\aligned"
IS_END = "2021-12-31"

PAIRS = ["002463_002841", "002841_600363", "002841_300458", "002371_002841",
         "002475_002841", "002736_601881", "002841_300433", "600061_601881",
         "600369_601555", "002407_600160", "002841_600601", "000783_601881",
         "002841_300136", "601555_601881", "600369_601881", "600977_601098"]


def eg_beta(la, lb):
    X = np.column_stack([np.ones(len(lb)), lb])
    b, *_ = np.linalg.lstsq(X, la, rcond=None)
    return b[1], b[0]          # (beta, alpha)


def hl_of(resid):
    rho = np.dot(resid[:-1], resid[1:]) / np.dot(resid[:-1], resid[:-1])
    if abs(rho) <= 0 or abs(rho) == 1:
        return np.nan, rho
    return -np.log(2.0) / np.log(abs(rho)), rho


def main() -> None:
    print(f"{'pair':<14} {'beta_is':>8} {'beta_oos':>8} {'drift%':>7} "
          f"{'ADF_oos(beta_is)':>16} {'ADF_oos(beta_oos)':>16} {'hl_is':>7} {'hl_oos(beta_oos)':>16} {'结论'}")
    for pr in PAIRS:
        m = pd.read_parquet(f"{ALIGNED}\\{pr}.parquet")
        m = m.dropna(subset=["close_a", "close_b"])
        la = np.log(m["close_a"].to_numpy(float))
        lb = np.log(m["close_b"].to_numpy(float))
        mask = m["date"].to_numpy() <= np.datetime64(IS_END)
        la_is, lb_is = la[mask], lb[mask]
        la_oo, lb_oo = la[~mask], lb[~mask]

        b_is, a_is = eg_beta(la_is, lb_is)
        b_oo, a_oo = eg_beta(la_oo, lb_oo)
        drift = abs(b_oo - b_is) / max(abs(b_is), 1e-9) * 100

        # OOS 残差: 用 IS β vs 用 OOS β
        s1 = la_oo - (b_is * lb_oo + a_is)
        s2 = la_oo - (b_oo * lb_oo + a_oo)
        p1 = adfuller(s1, autolag="AIC")[1]
        p2 = adfuller(s2, autolag="AIC")[1]
        hl_is, _ = hl_of(la_is - (b_is * lb_is + a_is))
        hl_oo, _ = hl_of(s2)

        if p2 < 0.05:
            verdict = "β漂移假失败(卡尔曼可救)"
        else:
            verdict = "关系真破裂(静态不成立)"
        print(f"{pr:<14} {b_is:>8.3f} {b_oo:>8.3f} {drift:>6.0f}% "
              f"{p1:>16.2e} {p2:>16.2e} {hl_is:>7.1f} {hl_oo:>16.1f} {verdict}")


if __name__ == "__main__":
    main()
