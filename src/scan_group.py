"""core3 扫描按 z_entry 分组统计。"""
import pandas as pd

df = pd.read_csv(r"F:\Deepseekwork\秋招\project2_pairs\reports\sensitivity_core3.csv",
                 encoding="utf-8-sig")
print("== 按 z_entry 分组 ==")
for ze, g in df.groupby("z_entry"):
    pos = int((g["net_ret"] > 0).sum())
    print(f"  z_entry={ze}: 正收益 {(pos)}/{len(g)} 格 | "
          f"net_ret 中位数 {g['net_ret'].median():+.4f} | 夏普中位数 {g['sharpe'].median():+.2f}")
print()
print("== z_entry=2.5 全部格子 ==")
sub = df[df["z_entry"] == 2.5][["z_exit", "z_stop", "w0", "net_ret", "annual", "sharpe", "dd", "n_trades"]]
print(sub.to_string(index=False))
