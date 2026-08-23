r"""容量初估 v2：真实 ADV + 执行金额分布。"""
import os

import pandas as pd

REPORTS = r"F:\Deepseekwork\秋招\project2_pairs\reports"
BT = r"F:\Deepseekwork\秋招\project2_pairs\data\backtest"
P1 = r"F:\Deepseekwork\秋招\project1_factor\data\raw\qfq"

ex = pd.read_csv(os.path.join(REPORTS, "final_exec.csv"), encoding="utf-8-sig")
ex["date"] = pd.to_datetime(ex["date"])

# 真实 ADV（2022-2026 平均日成交额）
adv = {}
for code in ex["code"].astype(str).str.zfill(6).unique():
    df = pd.read_parquet(os.path.join(P1, f"{code}.parquet"), columns=["date", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2022-01-01"]
    adv[code] = df["amount"].mean()

print("== 标的真实日均成交额（2022-2026）==")
for c, a in sorted(adv.items(), key=lambda x: -x[1]):
    print(f"  {c}: {a/1e8:.2f} 亿")

print("\n== 执行金额分布（NAV 单位）==")
print(ex.groupby(ex["code"].astype(str).str.zfill(6))["amount"].describe()[["mean", "max", "count"]].to_string())

print("\n== 容量表（单笔 = P99 执行金额 × 规模；冲击比 = 单笔/ADV）==")
for code, a in adv.items():
    g = ex[ex["code"].astype(str).str.zfill(6) == code]
    p99 = g["amount"].quantile(0.99)
    for cap in [1e8, 5e8, 1e9]:
        trade = p99 * cap
        print(f"  {code}: 规模{cap/1e8:.0f}亿 -> 单笔{trade/1e4:.0f}万 = ADV 的 {trade/a:.1%}")
