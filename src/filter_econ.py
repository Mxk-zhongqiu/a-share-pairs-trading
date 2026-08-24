"""从扩展池选择集中剔除 β 符号翻转对（经济逻辑过滤，预注册 UPGRADE.md P0-2c）。"""
import pandas as pd

sel = pd.read_csv(r"F:\Deepseekwork\秋招\project2_pairs\data\pairs\coint_selected.csv",
                  encoding="utf-8-sig")
flip = sel["beta_is"] * sel["beta_oos"] < 0
print(f"过滤前 {len(sel)} 对, β 符号翻转 {int(flip.sum())} 对:")
print(sel[flip]["pair"].to_string(index=False))
sel = sel[~flip].reset_index(drop=True)
sel.to_csv(r"F:\Deepseekwork\秋招\project2_pairs\data\pairs\coint_selected.csv",
           index=False, encoding="utf-8-sig")
print(f"过滤后 {len(sel)} 对 -> coint_selected.csv")
