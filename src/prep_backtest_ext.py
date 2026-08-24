r"""扩展池回测输入准备：close/limit 矩阵（多数据根）+ 涨跌停从价格计算。

涨跌停口径（与项目一一致，收盘阈值近似）:
    主板(60/00): ±10%；创业板(300/301): 2020-08-24 前 ±10% 后 ±20%；
    科创板(688): ±20%。|日收益| ≥ 阈值-0.002 标记。
输出: data/backtest_ext/{close_matrix,limit_up_matrix,limit_down_matrix,pair_list}.parquet/csv
"""
import os

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(PROJ2, "data", "backtest_ext")
P1 = r"F:\Deepseekwork\秋招\project1_factor\data"
RAW_EXT = os.path.join(PROJ2, "data", "raw_ext")
FIX = os.path.join(PROJ2, "data", "raw_fixed", "qfq")
CHINEXT_REFORM = pd.Timestamp("2020-08-24")


def find_qfq(code: str) -> str | None:
    for root in (FIX, os.path.join(P1, "raw", "qfq"), os.path.join(RAW_EXT, "qfq")):
        p = os.path.join(root, f"{code}.parquet")
        if os.path.exists(p):
            return p
    return None


def limit_pct(code: str, d: pd.Timestamp) -> float:
    if code.startswith("688"):
        return 0.20
    if code.startswith(("300", "301")):
        return 0.10 if d < CHINEXT_REFORM else 0.20
    return 0.10


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    sel = pd.read_csv(os.path.join(PROJ2, "data", "pairs", "coint_selected.csv"),
                      encoding="utf-8-sig")
    codes = sorted(set(sel["pair"].str.split("_").str[0]) | set(sel["pair"].str.split("_").str[1]))
    print(f"[pairs] {len(sel)} 对 | {len(codes)} 只")

    closes, ups, downs = {}, {}, {}
    for c in codes:
        p = find_qfq(c)
        if p is None:
            raise FileNotFoundError(c)
        df = pd.read_parquet(p, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")["close"].rename(c)
        ret = df.pct_change()
        up = pd.Series(False, index=df.index)
        down = pd.Series(False, index=df.index)
        for d, r in ret.items():
            if pd.isna(r):
                continue
            pct = limit_pct(c, d)
            if r >= pct - 0.002:
                up[d] = True
            elif r <= -(pct - 0.002):
                down[d] = True
        closes[c] = df
        up.name = c
        down.name = c
        ups[c] = up
        downs[c] = down
        if len(codes) <= 3 or c in ("601198", "601375", "000767"):
            print(f"  [{c}] rows={len(df)} up={int(up.sum())} down={int(down.sum())}")

    close_df = pd.concat(closes.values(), axis=1).sort_index()
    up_df = pd.concat(ups.values(), axis=1).reindex(index=close_df.index, columns=codes)
    down_df = pd.concat(downs.values(), axis=1).reindex(index=close_df.index, columns=codes)
    up_df = up_df.fillna(False).astype(bool)
    down_df = down_df.fillna(False).astype(bool)

    close_df.to_parquet(os.path.join(OUT, "close_matrix.parquet"))
    up_df.to_parquet(os.path.join(OUT, "limit_up_matrix.parquet"))
    down_df.to_parquet(os.path.join(OUT, "limit_down_matrix.parquet"))
    sel.to_csv(os.path.join(OUT, "pair_list.csv"), index=False, encoding="utf-8-sig")
    print(f"[limits] 涨停日合计 {int(up_df.sum().sum())} 跌停 {int(down_df.sum().sum())}")
    print(f"[close] {close_df.shape} | DONE -> {OUT}")


if __name__ == "__main__":
    main()
