r"""扩展候选池构建：hs800（项目一）+ 中证1000 新增（raw_ext）并集。

过滤规则与 CANDIDATE_POOL.md 一致（预注册不变）：
申万一级行业 / 日均成交额≥1亿 / 剔 ST / 缺失≤40 交易日 / 行业≥3 只。
数据加载：优先项目一（hs800），其次 raw_ext（CSI1000 新增）。
"""
import os
import time

import akshare as ak
import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(PROJ2, "data", "pairs")
P1_DATA = r"F:\Deepseekwork\秋招\project1_factor\data"
RAW_EXT = os.path.join(PROJ2, "data", "raw_ext")
START, END = "2017-01-01", "2026-08-21"


def load_close(code: str) -> pd.DataFrame | None:
    """按 项目一 → raw_ext 顺序找 qfq 文件。"""
    p1 = os.path.join(P1_DATA, "raw", "qfq", f"{code}.parquet")
    ext = os.path.join(RAW_EXT, "qfq", f"{code}.parquet")
    p = p1 if os.path.exists(p1) else (ext if os.path.exists(ext) else None)
    if p is None:
        return None
    return pd.read_parquet(p, columns=["date", "amount"])


def sw_map(codes_all: set) -> pd.DataFrame:
    first = ak.sw_index_first_info()
    rows = []
    t0 = time.time()
    for i, row in first.iterrows():
        sym = str(row.iloc[0])
        ind = str(row.iloc[1])
        code6 = sym.split(".")[0]
        try:
            cons = ak.index_component_sw(symbol=code6)
            col = [c for c in cons.columns if "代码" in c or "code" in c.lower()]
            for c in cons[col[0]].astype(str).str.zfill(6):
                if c in codes_all:
                    rows.append({"code": c, "sw_industry": ind})
        except Exception:
            pass
        time.sleep(0.3)
    mp = pd.DataFrame(rows).drop_duplicates("code", keep="first")
    print(f"[SW] 命中 {len(mp)}/{len(codes_all)} | {(time.time()-t0)/60:.1f}min")
    return mp


def official_calendar() -> set:
    cal = ak.tool_trade_date_hist_sina()
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    cal = cal[(cal["trade_date"] >= START) & (cal["trade_date"] <= END)]
    return set(cal["trade_date"])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    # 全量候选代码：hs800 + CSI1000
    uni = pd.read_parquet(os.path.join(P1_DATA, "raw", "universe.parquet"))
    hs800 = set(uni["code"].astype(str).str.zfill(6))
    ext = pd.read_csv(os.path.join(PROJ2, "data", "pairs", "universe_csi1000.csv"),
                      encoding="utf-8-sig")
    all_codes = sorted(hs800 | set(ext["code"].astype(str).str.zfill(6)))
    print(f"[universe] hs800 {len(hs800)} + CSI1000 新增 = {len(all_codes)}")

    # 申万一级行业（全量）
    ind = sw_map(set(all_codes))

    # 官方日历
    cal = official_calendar()

    # 逐股票统计（流动性/缺失）
    stats = []
    for i, code in enumerate(all_codes, 1):
        df = load_close(code)
        if df is None:
            continue
        df["date"] = pd.to_datetime(df["date"])
        dates = set(df["date"])
        stats.append({"code": code, "n_days": len(df),
                      "missing_days": len(cal - dates),
                      "avg_amount": float(np.nanmean(df["amount"].to_numpy())),
                      "median_amount": float(np.nanmedian(df["amount"].to_numpy()))})
        if i % 300 == 0:
            print(f"  [stats] {i}/{len(all_codes)}")
    stats_df = pd.DataFrame(stats)

    # ST
    st_path = os.path.join(P1_DATA, "processed", "st_list.parquet")
    st_set = set(pd.read_parquet(st_path)["code"].astype(str).str.zfill(6))

    merged = stats_df.merge(ind, on="code", how="left")
    merged["is_st"] = merged["code"].isin(st_set)
    before = len(merged)
    merged = merged[
        (merged["sw_industry"].notna())
        & (~merged["is_st"])
        & (merged["avg_amount"] >= 1e8)
        & (merged["missing_days"] <= 40)
    ].copy()
    print(f"[filter] {before} -> {len(merged)} 只（行业/ST/流动性/缺失）")
    merged.to_parquet(os.path.join(OUT, "pool_filtered_ext.parquet"), index=False)

    # 候选对（同行业两两，行业≥3只）
    pair_rows = []
    for ind_name, grp in merged.groupby("sw_industry"):
        codes = sorted(grp["code"])
        if len(codes) < 3:
            continue
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                pair_rows.append({"industry": ind_name,
                                  "code_a": codes[i], "code_b": codes[j]})
    pairs = pd.DataFrame(pair_rows)
    print(f"[pairs] 候选对 {len(pairs)}")
    pairs.to_parquet(os.path.join(OUT, "candidate_pairs_ext.parquet"), index=False)

    summ = merged.groupby("sw_industry").agg(n=("code", "count")).sort_values("n", ascending=False)
    print(summ.head(15).to_string())
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
