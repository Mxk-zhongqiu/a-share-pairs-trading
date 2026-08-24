r"""协整对最终选择 v2（三层协议，预注册见 docs/UPGRADE.md P0-2b）。

tier1 IS 筛选:     BH-FDR(q=0.05) over m=2×N（双向 EG min p, IS 2017-2021）
tier2 确认窗:      BH-FDR(q=0.05) over tier1（重估 β 的残差 ADF p, 2022-2023）
                    + hl_oos∈[0,120]
tier3 方向1过滤:   交易方向（la~lb）IS p 过 BH-FDR(q=0.05) over m=N（全对方向1 p）
输出:
    coint_confirmed.csv    tier1+tier2（确认对）
    coint_selected.csv     三层全过（可交易核心对）
"""
import os

import numpy as np
import pandas as pd

PAIRS = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")), "data", "pairs")
Q = 0.05


def bh_pass(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR: 返回按原顺序的通过布尔掩码。"""
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    n = len(sorted_p)
    passed_sorted = sorted_p < (np.arange(1, n + 1) / n) * Q
    ok_idx = np.where(passed_sorted)[0]
    mask = np.zeros(n, dtype=bool)
    if len(ok_idx):
        k = ok_idx.max()
        mask[order[:k + 1]] = True
    return mask


def main() -> None:
    df = pd.read_csv(os.path.join(PAIRS, "coint_results.csv"), encoding="utf-8-sig")
    d1 = pd.read_csv(os.path.join(PAIRS, "dir1_p_is.csv"), encoding="utf-8-sig")
    df = df.merge(d1[["pair", "dir1_p_is"]], on="pair", how="left")
    N = len(df)
    print(f"[total] {N} 对 | 确认窗口 2022-2023（交易期 2024+ 不参与选择）")

    # tier1: IS-FDR（双向 min p, m=2N）
    df["pass_t1"] = bh_pass(df["min_p_is"].to_numpy(float))
    t1 = df[df["pass_t1"]].copy()
    print(f"[tier1] IS-FDR: {len(t1)} 对")

    # tier2: 确认窗 FDR + hl 门
    t1 = t1.sort_values("p_oos_rees")
    t1["pass_t2"] = bh_pass(t1["p_oos_rees"].to_numpy(float))
    t1["hl_ok"] = (t1["hl_oos"] >= 0) & (t1["hl_oos"] <= 120)
    t1["confirmed"] = t1["pass_t2"] & t1["hl_ok"]
    conf = t1[t1["confirmed"]].copy()
    print(f"[tier2] 确认窗 FDR+hl: {len(conf)} 对")

    # tier3: 方向1 IS p 过全对 FDR(m=N)
    t3_pass = bh_pass(d1["dir1_p_is"].to_numpy(float))
    t3_set = set(d1.loc[t3_pass, "pair"])
    conf["pass_t3"] = conf["pair"].isin(t3_set)
    core = conf[conf["pass_t3"]].sort_values("p_oos_rees")
    print(f"[tier3] 方向1 IS-FDR: 可交易核心 {len(core)} 对")

    conf.to_csv(os.path.join(PAIRS, "coint_confirmed.csv"), index=False, encoding="utf-8-sig")
    core.to_csv(os.path.join(PAIRS, "coint_selected.csv"), index=False, encoding="utf-8-sig")
    print("\n== 可交易核心对 ==")
    if len(core):
        print(core[["pair", "min_p_is", "dir1_p_is", "hl_is", "p_oos_rees",
                    "hl_oos", "beta_is", "beta_drift"]].to_string(index=False))
    else:
        print("（空——三层协议无一通过，诚实报告）")
    print(f"\nDONE -> {PAIRS}")


if __name__ == "__main__":
    main()
