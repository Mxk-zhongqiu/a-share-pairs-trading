r"""扩展池双股对齐 QA（多数据根：项目一 → raw_ext → 修复副本 优先级）。

检查项与 qa_alignment.py 一致（预注册口径不变）：
共同交易日≥1500 / 复权一致性抽查 / 极端收益>21% / 零价。
输出: data/aligned_ext/<code_a>_<code_b>.parquet + QA 汇总。
"""
import os

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAIRS = os.path.join(PROJ2, "data", "pairs")
ALIGNED_OUT = os.path.join(PROJ2, "data", "aligned_ext")
P1 = r"F:\Deepseekwork\秋招\project1_factor\data"
RAW_EXT = os.path.join(PROJ2, "data", "raw_ext")
FIX = os.path.join(PROJ2, "data", "raw_fixed", "qfq")
MIN_COMMON = 1500
EXTREME = 0.21
RET_DIFF = 0.05


def find_qfq(code: str) -> str | None:
    for root in (FIX, os.path.join(P1, "raw", "qfq"), os.path.join(RAW_EXT, "qfq")):
        p = os.path.join(root, f"{code}.parquet")
        if os.path.exists(p):
            return p
    return None


def find_raw(code: str) -> str | None:
    p1 = os.path.join(P1, "raw", "raw", f"{code}.parquet")
    ext = os.path.join(RAW_EXT, "raw", f"{code}.parquet")
    return p1 if os.path.exists(p1) else (ext if os.path.exists(ext) else None)


def check_qfq_raw(code: str, dates: pd.DatetimeIndex) -> dict:
    qp, rp = find_qfq(code), find_raw(code)
    if qp is None or rp is None:
        return {"n_bad": np.nan, "max_diff": np.nan}
    q = pd.read_parquet(qp, columns=["date", "close"])
    r = pd.read_parquet(rp, columns=["date", "close"])
    q["date"] = pd.to_datetime(q["date"]); r["date"] = pd.to_datetime(r["date"])
    m = q.merge(r, on="date", suffixes=("_q", "_r"))
    m = m[m["date"].isin(dates)]
    m["rq"] = m["close_q"].pct_change(); m["rr"] = m["close_r"].pct_change()
    both = m[(m["rq"].abs() > 1e-9) & (m["rr"].abs() > 1e-9) & (m["rq"].abs() < 0.095)]
    if len(both) < 10:
        return {"n_bad": np.nan, "max_diff": np.nan}
    idx = np.linspace(0, len(both) - 1, 8).astype(int)
    d = (both.iloc[idx]["rq"] - both.iloc[idx]["rr"]).abs()
    return {"n_bad": int((d > RET_DIFF).sum()), "max_diff": float(d.max())}


def main() -> None:
    os.makedirs(ALIGNED_OUT, exist_ok=True)
    pairs = pd.read_parquet(os.path.join(PAIRS, "candidate_pairs_ext.parquet"))
    print(f"[pairs] {len(pairs)} 对待 QA")

    rows, n_pass = [], 0
    for i, (_, pr) in enumerate(pairs.iterrows(), 1):
        ca, cb = pr["code_a"], pr["code_b"]
        qa_p, qb_p = find_qfq(ca), find_qfq(cb)
        if qa_p is None or qb_p is None:
            continue
        da = pd.read_parquet(qa_p, columns=["date", "close"]); da["date"] = pd.to_datetime(da["date"])
        db = pd.read_parquet(qb_p, columns=["date", "close"]); db["date"] = pd.to_datetime(db["date"])
        m = da.merge(db, on="date", suffixes=("_a", "_b")).sort_values("date").reset_index(drop=True)
        n_common = len(m)
        if n_common < MIN_COMMON:
            rows.append({"code_a": ca, "code_b": cb, "n_common": n_common, "qa_pass": False})
            continue
        m["ret_a"] = m["close_a"].pct_change(); m["ret_b"] = m["close_b"].pct_change()
        bad = int(((m["close_a"] <= 0) | (m["close_b"] <= 0)).sum())
        extreme = int(((m["ret_a"].abs() > EXTREME) | (m["ret_b"].abs() > EXTREME)).sum())
        qaA = check_qfq_raw(ca, pd.DatetimeIndex(m["date"]))
        qaB = check_qfq_raw(cb, pd.DatetimeIndex(m["date"]))
        ok = (bad == 0 and extreme <= 3
              and (qaA["n_bad"] == 0 or np.isnan(qaA["n_bad"]))
              and (qaB["n_bad"] == 0 or np.isnan(qaB["n_bad"])))
        rows.append({"code_a": ca, "code_b": cb, "n_common": n_common,
                     "bad_price": bad, "extreme_ret": extreme,
                     "qa_a": qaA["n_bad"], "qa_b": qaB["n_bad"], "qa_pass": ok})
        if ok:
            out = m[["date", "close_a", "close_b", "ret_a", "ret_b"]].copy()
            out.to_parquet(os.path.join(ALIGNED_OUT, f"{ca}_{cb}.parquet"), index=False)
            n_pass += 1
        if i % 2000 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] pass={n_pass}")

    qa = pd.DataFrame(rows)
    qa.to_csv(os.path.join(PAIRS, "alignment_qa_ext.csv"), index=False, encoding="utf-8-sig")
    print(f"DONE: {len(qa)} 对, QA 通过 {n_pass} ({n_pass/len(qa):.1%})")


if __name__ == "__main__":
    main()
