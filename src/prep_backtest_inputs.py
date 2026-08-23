r"""回测输入准备（p4b）：从 9 个核心对构建引擎输入矩阵。

输出 (data/backtest/):
    close_matrix.parquet     date × code qfq 收盘（修复副本优先）
    limit_up_matrix.parquet  date × code 涨停标记
    limit_down_matrix.parquet date × code 跌停标记
    pair_list.csv            交易对清单（含 code_a/code_b）
"""
import os

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAIRS = os.path.join(PROJ2, "data", "pairs")
BT = os.path.join(PROJ2, "data", "backtest")
P1_DATA = r"F:\Deepseekwork\秋招\project1_factor\data"
FIX_OVERLAY = os.path.join(PROJ2, "data", "raw_fixed", "qfq")


def main() -> None:
    os.makedirs(BT, exist_ok=True)
    sel = pd.read_csv(os.path.join(PAIRS, "coint_selected.csv"), encoding="utf-8-sig")
    codes = sorted(set(sel["pair"].str.split("_").str[0]) | set(sel["pair"].str.split("_").str[1]))
    print(f"[pairs] {len(sel)} 对 | 涉及 {len(codes)} 只股票")

    # close 矩阵（qfq，修复副本优先）
    closes = {}
    for c in codes:
        if os.path.exists(os.path.join(FIX_OVERLAY, f"{c}.parquet")):
            df = pd.read_parquet(os.path.join(FIX_OVERLAY, f"{c}.parquet"), columns=["date", "close"])
        else:
            df = pd.read_parquet(os.path.join(P1_DATA, "raw", "qfq", f"{c}.parquet"),
                                 columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        closes[c] = df.set_index("date")["close"].rename(c)
    close_df = pd.concat(closes.values(), axis=1).sort_index()
    print(f"[close] {close_df.shape[0]} 日 × {close_df.shape[1]} 股")

    # 涨跌停矩阵
    lim = pd.read_parquet(os.path.join(P1_DATA, "processed", "limits.parquet"))
    lim = lim[lim["code"].isin(codes)]
    lim["date"] = pd.to_datetime(lim["date"])
    up = lim.pivot(index="date", columns="code", values="limit_up").sort_index()
    down = lim.pivot(index="date", columns="code", values="limit_down").sort_index()
    up = up.reindex(index=close_df.index, columns=codes)
    down = down.reindex(index=close_df.index, columns=codes)
    print(f"[limits] up {up.shape} down {down.shape} | 涨停日数合计 {int(up.sum().sum())}")

    close_df.to_parquet(os.path.join(BT, "close_matrix.parquet"))
    up.to_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down.to_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    sel.to_csv(os.path.join(BT, "pair_list.csv"), index=False, encoding="utf-8-sig")
    print(f"DONE -> {BT}")


if __name__ == "__main__":
    main()
