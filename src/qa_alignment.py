r"""双股价格对齐 + QA（项目二新增模块，p2 阶段）。

用途:
    配对交易的一切都建立在"两只股票的价格序列在同一口径、同一日历上可比"。
    本脚本把候选对的两只股票 qfq 收盘价对齐到共同交易日，并做数据质量检查
    （项目一 P1-P3 教训平移 + 项目二新增的双股对齐检查）。

QA 项:
    a) 交易日历: 共同交易日数（过少 => 该对数据质量不足，剔除）；
       停牌缺口（单边缺失区间）显式记录
    b) 复权一致性 spot-check: 每对抽查若干非除权日，qfq 收益 vs raw 收益
       |diff| > 5% => 复权 bug 预警（P1-P3）
    c) 价格异常: close<=0 / 极端单日收益（P1 零价）

输出:
    data/pairs/alignment_qa.csv       每对 QA 汇总
    data/aligned/<code_a>_<code_b>.parquet   对齐后的价格序列（通过 QA 的对）
        列: date, close_a, close_b, ret_a, ret_b, gap_a, gap_b
        gap_* = 该股当日是否停牌缺失（0/1）
"""
import argparse
import os

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAIRS = os.path.join(PROJ2, "data", "pairs")
ALIGNED = os.path.join(PROJ2, "data", "aligned")

RET_DIFF_THRESHOLD = 0.05   # qfq vs raw 单日收益差异阈值
MIN_COMMON_DAYS = 1500      # 共同交易日下限（2017-2026 共 2340 日，取 ~64%）
ZERO_PRICE_TOL = 1e-9


def load_close(data_root: str, code: str, adjust: str, overlay: str | None = None) -> pd.DataFrame:
    """读取日线；overlay 目录存在对应文件时优先（项目二修复副本优先于项目一原数据）。"""
    if overlay and adjust == "qfq" and os.path.exists(os.path.join(overlay, f"{code}.parquet")):
        df = pd.read_parquet(os.path.join(overlay, f"{code}.parquet"),
                             columns=["date", "close", "amount"])
    else:
        df = pd.read_parquet(os.path.join(data_root, "raw", adjust, f"{code}.parquet"),
                             columns=["date", "close", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def check_qfq_vs_raw(data_root: str, code: str, dates: pd.DatetimeIndex,
                     n_sample: int = 8, overlay: str | None = None) -> dict:
    """抽查非除权日 qfq 收益 vs raw 收益一致性。返回 {n_bad, n_checked, max_diff}。"""
    q = load_close(data_root, code, "qfq", overlay)
    r = load_close(data_root, code, "raw")
    m = q.merge(r, on="date", suffixes=("_q", "_r"))
    m = m[m["date"].isin(dates)]
    if len(m) < 10:
        return {"n_bad": np.nan, "n_checked": len(m), "max_diff": np.nan}
    m["rq"] = m["close_q"].pct_change()
    m["rr"] = m["close_r"].pct_change()
    # 非除权日: 两收益都非零且差异显著才可能是 bug；除权日 raw 会跳变
    both = m[(m["rq"].abs() > 1e-9) & (m["rr"].abs() > 1e-9)]
    both = both[both["rq"].abs() < 0.095]   # 去掉涨跌停日（近似）
    if len(both) == 0:
        return {"n_bad": np.nan, "n_checked": len(both), "max_diff": np.nan}
    idx = np.linspace(0, len(both) - 1, min(n_sample, len(both))).astype(int)
    sample = both.iloc[idx]
    diff = (sample["rq"] - sample["rr"]).abs()
    return {"n_bad": int((diff > RET_DIFF_THRESHOLD).sum()),
            "n_checked": len(sample),
            "max_diff": float(diff.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"F:\Deepseekwork\秋招\project1_factor\data")
    ap.add_argument("--overlay", default=None,
                    help="修复后 qfq 目录（data/raw_fixed/qfq），存在时优先读取")
    ap.add_argument("--max-pairs", type=int, default=0, help="仅处理前 N 对（调试）")
    args = ap.parse_args()

    os.makedirs(ALIGNED, exist_ok=True)
    pairs = pd.read_parquet(os.path.join(PAIRS, "candidate_pairs.parquet"))
    if args.max_pairs:
        pairs = pairs.head(args.max_pairs)

    # 一次性加载全市场 qfq close（463 只 × 2340 行，约 40MB，比逐对重读快）
    codes = sorted(set(pairs["code_a"]) | set(pairs["code_b"]))
    closes = {}
    for c in codes:
        closes[c] = load_close(args.data_root, c, "qfq", args.overlay)

    rows = []
    n_pass = 0
    for i, (_, pr) in enumerate(pairs.iterrows(), 1):
        ca, cb = pr["code_a"], pr["code_b"]
        da, db = closes[ca], closes[cb]
        m = da.merge(db, on="date", suffixes=("_a", "_b"))
        m = m.sort_values("date").reset_index(drop=True)
        n_common = len(m)
        if n_common < MIN_COMMON_DAYS:
            rows.append({"code_a": ca, "code_b": cb, "n_common": n_common,
                         "qa_pass": False, "reason": "共同交易日不足"})
            continue

        # 停牌缺口：单边缺失（用全量日期集合算）
        all_dates = set(da["date"]) | set(db["date"])
        gap_a = len(all_dates - set(da["date"]))
        gap_b = len(all_dates - set(db["date"]))

        # 价格异常 / 极端收益
        bad_price = int(((m["close_a"] <= ZERO_PRICE_TOL) | (m["close_b"] <= ZERO_PRICE_TOL)).sum())
        m["ret_a"] = m["close_a"].pct_change()
        m["ret_b"] = m["close_b"].pct_change()
        extreme = int(((m["ret_a"].abs() > 0.21) | (m["ret_b"].abs() > 0.21)).sum())

        # 复权一致性（两只都抽查）
        qa_a = check_qfq_vs_raw(args.data_root, ca, pd.DatetimeIndex(m["date"]), overlay=args.overlay)
        qa_b = check_qfq_vs_raw(args.data_root, cb, pd.DatetimeIndex(m["date"]), overlay=args.overlay)

        qa_ok = (bad_price == 0 and extreme <= 3
                 and (qa_a["n_bad"] == 0 or np.isnan(qa_a["n_bad"]))
                 and (qa_b["n_bad"] == 0 or np.isnan(qa_b["n_bad"])))
        rows.append({
            "code_a": ca, "code_b": cb,
            "n_common": n_common, "gap_a": gap_a, "gap_b": gap_b,
            "bad_price": bad_price, "extreme_ret": extreme,
            "qfqraw_bad_a": qa_a["n_bad"], "qfqraw_bad_b": qa_b["n_bad"],
            "max_diff_a": qa_a["max_diff"], "max_diff_b": qa_b["max_diff"],
            "qa_pass": qa_ok,
            "reason": "" if qa_ok else f"bad_price={bad_price},extreme={extreme},qa_a={qa_a['n_bad']},qa_b={qa_b['n_bad']}",
        })

        if qa_ok:
            out = m[["date", "close_a", "close_b", "ret_a", "ret_b"]].copy()
            out["gap_a"] = (~out["date"].isin(da["date"])).astype(int)
            out["gap_b"] = (~out["date"].isin(db["date"])).astype(int)
            out.to_parquet(os.path.join(ALIGNED, f"{ca}_{cb}.parquet"), index=False)
            n_pass += 1

        if i % 500 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] pass={n_pass}")

    qa = pd.DataFrame(rows)
    qa.to_csv(os.path.join(PAIRS, "alignment_qa.csv"), index=False, encoding="utf-8-sig")
    print(f"DONE: 共 {len(qa)} 对, QA 通过 {n_pass} 对, 通过率 {n_pass/len(qa):.1%}")
    print("未通过原因分布:")
    print(qa[~qa["qa_pass"]]["reason"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
