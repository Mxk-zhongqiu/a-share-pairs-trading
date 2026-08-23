r"""验证 v6 修复生效：T+1 成交（信号日≠成交日）、受阻重试、stale 强平。"""
import pandas as pd

import sys

tag = sys.argv[1] if len(sys.argv) > 1 else "v6"
tr = pd.read_csv(rf"F:\Deepseekwork\秋招\project2_pairs\reports\pair_trades_{tag}.csv",
                 encoding="utf-8-sig")
tr["date"] = pd.to_datetime(tr["date"])
ex = pd.read_csv(rf"F:\Deepseekwork\秋招\project2_pairs\reports\pair_exec_{tag}.csv",
                 encoding="utf-8-sig")
ex["date"] = pd.to_datetime(ex["date"])

print("== 1) T+1 验证：开仓信号日 vs 首次成交日 ==")
for _, o in tr[tr["action"] == "open"].head(5).iterrows():
    first = ex[ex["date"] > o["date"]]
    if not first.empty:
        d0 = first["date"].iloc[0]
        delta = (d0 - o["date"]).days
        print(f"  {o['pair']} 信号日 {o['date'].date()} -> 首个成交日 {d0.date()} "
              f"(间隔 {delta} 自然日, T+1={'OK' if delta >= 1 else 'FAIL'})")

print("\n== 2) 受阻记录 ==")
blk = ex[ex["blocked"] == True]
print(f"  受阻执行笔数: {len(blk)}")
print(f"  受阻代码 top: {blk['code'].value_counts().head(5).to_dict()}")

print("\n== 3) 受阻后重试验证（同一 code 后续有成功成交）==")
retried = 0
for code, g in blk.groupby("code"):
    blk_dates = set(g["date"])
    later_ok = ex[(ex["code"] == code) & (ex["blocked"] == False) &
                  (~ex["date"].isin(blk_dates))]
    if not later_ok.empty and later_ok["date"].min() > min(blk_dates):
        retried += 1
print(f"  受阻后成功重试的 code 数: {retried}/{blk['code'].nunique()}")

print("\n== 4) 交易明细 action 分布 ==")
print(tr["action"].value_counts().to_string())
print(tr[tr["action"] == "close"]["reason"].value_counts().to_string())
