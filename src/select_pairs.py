r"""协整对最终选择（后处理，读取 coint_results.csv 免重跑）——两级 FDR 协议 v3。

协议 v3（预注册，见 docs/COINTEGRATION.md）:
    tier1 IS 筛选: BH-FDR(q=0.05) over m=2×N（全部双向测试）→ 候选集
    tier2 OOS 确认: BH-FDR(q=0.05) over tier1 + hl_oos∈[0,120] → 核心交易对
    严格对照: Bonferroni-IS ∩ OOS-FDR → 严格对（记录用）

输出:
    data/pairs/coint_selected.csv        两级 FDR 核心对（策略输入）
    data/pairs/coint_selected_strict.csv 严格对照（Bonf）
    data/pairs/selection_ladder.csv      阶梯全表（每对 IS/OOS 统计量 + 分层标记）
"""
import os

import numpy as np
import pandas as pd

PAIRS = os.path.join(os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")), "data", "pairs")
Q = 0.05


def bh_pass(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR: 返回按原顺序的通过布尔掩码。pvals 为每对 min_p。"""
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    n = len(sorted_p)
    thresh = (np.arange(1, n + 1) / n) * Q
    passed_sorted = sorted_p < thresh
    # BH: 从最小开始，通过阈值链（最大 i 使 p_i <= q·i/m 以下全部通过）
    ok_idx = np.where(passed_sorted)[0]
    if len(ok_idx) == 0:
        return np.zeros(n, dtype=bool)
    k = ok_idx.max()
    mask = np.zeros(n, dtype=bool)
    mask[order[:k + 1]] = True
    return mask


def main() -> None:
    df = pd.read_csv(os.path.join(PAIRS, "coint_results.csv"), encoding="utf-8-sig")
    print(f"[total] {len(df)} 对")

    # tier1: IS FDR
    df["pass_is_fdr"] = bh_pass(df["min_p_is"].to_numpy(float))
    t1 = df[df["pass_is_fdr"]].copy()
    print(f"[tier1] IS-FDR: {len(t1)} 对")

    # tier2: OOS FDR over tier1 + hl gate
    t1 = t1.sort_values("p_oos_rees")
    t1["pass_oos_fdr"] = bh_pass(t1["p_oos_rees"].to_numpy(float))
    t1["hl_ok_oos"] = (t1["hl_oos"] >= 0) & (t1["hl_oos"] <= 120)
    t1["selected"] = t1["pass_oos_fdr"] & t1["hl_ok_oos"]
    core = t1[t1["selected"]].sort_values("p_oos_rees")
    print(f"[tier2] OOS-FDR 通过: {t1['pass_oos_fdr'].sum()} | +hl 门: {len(core)} 对（核心交易对）")

    # 严格对照: Bonf IS ∩ OOS FDR
    strict = t1[t1["pass_is"] & t1["pass_oos_fdr"] & t1["hl_ok_oos"]]
    print(f"[strict] Bonf-IS ∩ OOS-FDR: {len(strict)} 对（对照）")

    core.to_csv(os.path.join(PAIRS, "coint_selected.csv"), index=False, encoding="utf-8-sig")
    strict.to_csv(os.path.join(PAIRS, "coint_selected_strict.csv"), index=False, encoding="utf-8-sig")
    t1.to_csv(os.path.join(PAIRS, "selection_ladder.csv"), index=False, encoding="utf-8-sig")

    print("\n== 核心交易对 ==")
    print(core[["pair", "min_p_is", "hl_is", "p_oos_rees", "hl_oos",
                "beta_is", "beta_oos", "beta_drift"]].to_string(index=False))
    print(f"\nDONE -> {PAIRS}")


if __name__ == "__main__":
    main()
