r"""终版 NAV 图 + 月度收益（报告 §4 配图）。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJ2)
REPORTS = os.path.join(PROJ2, "reports")

nav = pd.read_csv(os.path.join(REPORTS, "final_nav.csv"), index_col=0, parse_dates=True)
nav.columns = ["nav"]
nav = nav[nav.index >= "2022-01-01"]

# 沪深300 基准（同期归一）
bench = pd.read_parquet(r"F:\Deepseekwork\秋招\project1_factor\data\processed\benchmark.parquet")
bench["date"] = pd.to_datetime(bench["date"])
bench = bench[(bench["date"] >= "2022-01-01")].set_index("date")
bcol = [c for c in bench.columns if "close" in c.lower() or "300" in c][0]
b = bench[bcol]
b = b / b.iloc[0]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(nav.index, nav.values, label="Pair Book (3-core, net)", lw=1.6)
ax.plot(b.index, b.values, label="HS300 (normalized)", lw=1.2, alpha=0.7)
ax.set_title("Pairs Trading: 3-pair direction-1 core, 2022-2026 OOS (net of costs)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "final_nav.png"), dpi=130)
plt.close(fig)
print("saved final_nav.png")

# 月度收益表
m = nav["nav"].resample("ME").last().pct_change().dropna()
print(f"月度: 均值 {m.mean():+.2%} 胜率 {(m > 0).mean():.0%} 最差 {m.min():.2%} 最好 {m.max():.2%}")
