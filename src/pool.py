r"""候选股票池构建（预注册定义见 docs/CANDIDATE_POOL.md，规则写死后不改）。

输入:
    --data-root     项目一数据根目录（默认 F:\Deepseekwork\秋招\project1_factor\data）
    --min-avg-amount  样本期日均成交额门槛（元，默认 1e8 = 1 亿）
    --min-members     行业内合格股票数下限（默认 3）
    --start / --end   与项目一数据区间一致（默认 2017-01-01 ~ 2026-08-21）

输出 (data/pairs/):
    universe_industry.parquet   # hs800 全量 × 申万一级行业归属
    pool_filtered.parquet       # 过滤后合格股票（含流动性统计）
    candidate_pairs.parquet     # 候选对（同行业两两组合，预注册用）
    pool_summary.csv            # 每行业: 原始数/合格数/候选对数

特性: 行业成分接口限速；逐股票只读 amount 列（快）；失败股票记录不中断。
"""
import argparse
import os
import time

import akshare as ak
import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(PROJ2, "data", "pairs")
START, END = "2017-01-01", "2026-08-21"


def load_universe(data_root: str) -> pd.DataFrame:
    uni = pd.read_parquet(os.path.join(data_root, "raw", "universe.parquet"))
    uni["code"] = uni["code"].astype(str).str.zfill(6)
    return uni


def fetch_sw_industry_map(uni_codes: set) -> pd.DataFrame:
    """申万一级（当前成分）→ 股票归属。返回 code→industry 映射表。"""
    first = ak.sw_index_first_info()          # 31 个申万一级
    rows = []
    fail = []
    t0 = time.time()
    for i, row in first.iterrows():
        sym = str(row.iloc[0])                 # 801010.SI
        ind = str(row.iloc[1])                 # 农林牧渔
        code6 = sym.split(".")[0]
        try:
            cons = ak.index_component_sw(symbol=code6)
            col_code = [c for c in cons.columns if "代码" in c or "code" in c.lower()]
            codes = cons[col_code[0]].astype(str).str.zfill(6)
            for c in codes:
                if c in uni_codes:
                    rows.append({"code": c, "sw_industry": ind})
        except Exception as e:
            fail.append((sym, str(e)[:80]))
        if (i + 1) % 8 == 0 or i == len(first) - 1:
            print(f"[SW] {i+1}/{len(first)} fail={len(fail)} elapsed={(time.time()-t0)/60:.1f}min")
        time.sleep(0.4)
    mp = pd.DataFrame(rows)
    if mp.empty:
        raise RuntimeError(f"申万成分映射为空, fail={fail}")
    dup = mp[mp.duplicated("code", keep=False)]
    if not dup.empty:
        print(f"[警告] {len(dup)} 只出现在多个申万一级行业，取第一个:", dup["code"].unique()[:10])
        mp = mp.drop_duplicates("code", keep="first")
    return mp


def official_calendar() -> set:
    cal = ak.tool_trade_date_hist_sina()
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    cal = cal[(cal["trade_date"] >= START) & (cal["trade_date"] <= END)]
    return set(cal["trade_date"])


def stock_stats(code: str, data_root: str, cal: set) -> dict | None:
    """读单只股票（只读 amount/date 列），返回统计；失败返回 None。"""
    p = os.path.join(data_root, "raw", "qfq", f"{code}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=["date", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    if df.empty:
        return None
    dates = set(df["date"])
    missing = len(cal - dates)
    return {
        "code": code,
        "n_days": len(df),
        "first_date": df["date"].min(),
        "last_date": df["date"].max(),
        "missing_days": missing,
        "avg_amount": float(np.nanmean(df["amount"].to_numpy())),
        "median_amount": float(np.nanmedian(df["amount"].to_numpy())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"F:\Deepseekwork\秋招\project1_factor\data")
    ap.add_argument("--min-avg-amount", type=float, default=1e8)
    ap.add_argument("--min-members", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    uni = load_universe(args.data_root)
    uni_codes = set(uni["code"])
    print(f"[universe] hs800 = {len(uni)} 只")

    # 1) 申万一级行业归属
    ind = fetch_sw_industry_map(uni_codes)
    print(f"[SW] 申万一级归属命中 {len(ind)}/{len(uni)} 只（未命中=已退市/更名/不在当前成分，披露为幸存者偏差）")

    # 2) 官方日历 + 逐股票流动性/缺失统计
    cal = official_calendar()
    print(f"[calendar] 官方交易日 {len(cal)} 天")
    stats = []
    for i, code in enumerate(sorted(uni_codes), 1):
        s = stock_stats(code, args.data_root, cal)
        if s is not None:
            stats.append(s)
        if i % 200 == 0:
            print(f"  [stats] {i}/{len(uni)}")
    stats_df = pd.DataFrame(stats)

    # 3) ST 标记（快照，披露）
    st_path = os.path.join(args.data_root, "processed", "st_list.parquet")
    st_set = set(pd.read_parquet(st_path)["code"].astype(str).str.zfill(6)) if os.path.exists(st_path) else set()

    # 4) 合并 + 过滤
    merged = uni.merge(ind, on="code", how="left").merge(stats_df, on="code", how="left")
    merged["is_st"] = merged["code"].isin(st_set)
    before = len(merged)
    merged = merged[
        (merged["sw_industry"].notna())
        & (~merged["is_st"])
        & (merged["avg_amount"] >= args.min_avg_amount)
        & (merged["missing_days"] <= 40)
    ].copy()
    print(f"[filter] {before} -> {len(merged)} 只（ST/无行业/流动性<{args.min_avg_amount:.0e}/缺失>40日 剔除）")

    merged.to_parquet(os.path.join(OUT, "pool_filtered.parquet"), index=False)

    # 5) 候选对生成（同行业两两，预注册全量）
    pair_rows = []
    for ind_name, grp in merged.groupby("sw_industry"):
        codes = sorted(grp["code"])
        if len(codes) < args.min_members:
            continue
        names = dict(zip(grp["code"], grp["name"]))
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                pair_rows.append({
                    "industry": ind_name,
                    "code_a": codes[i], "name_a": names[codes[i]],
                    "code_b": codes[j], "name_b": names[codes[j]],
                })
    pairs = pd.DataFrame(pair_rows)
    print(f"[pairs] 候选对总数 {len(pairs)}（同行业, 行业合格数≥{args.min_members}）")
    pairs.to_parquet(os.path.join(OUT, "candidate_pairs.parquet"), index=False)

    # 6) 汇总
    summary = merged.groupby("sw_industry").agg(
        n_stocks=("code", "count"),
    ).reset_index()
    summary["n_pairs"] = summary["n_stocks"].apply(lambda n: n * (n - 1) // 2 if n >= args.min_members else 0)
    summary = summary.sort_values("n_stocks", ascending=False)
    summary.to_csv(os.path.join(OUT, "pool_summary.csv"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
