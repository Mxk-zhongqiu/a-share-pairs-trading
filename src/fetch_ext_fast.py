r"""扩展池快取：新浪源（~4s/只，实测比腾讯 akshare 快 5 倍）+ 分块多进程 + 断点续传。

沙箱限制: multiprocessing 的 Pipe 被禁（WinError 5）→ 用"分块 + 多个独立进程"实现并行：
    每个进程处理一块代码（--n-chunks 等分，--chunk-id 指定块），各自断点续传。
    用法（启动 N 个后台任务）:
        python fetch_ext_fast.py --n-chunks 6 --chunk-id 0   # 进程 0
        python fetch_ext_fast.py --n-chunks 6 --chunk-id 1   # 进程 1
        ...
质量门槛: 行数>1800 / 无零价（对齐检查在 pool QA 阶段统一做）。
"""
import argparse
import os
import time

import akshare as ak
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(PROJ2, "data", "raw_ext")
START, END = "20170101", "20260821"


def to_sym(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def fetch_one(code: str, adjust: str) -> pd.DataFrame | None:
    """新浪日线，重试 3 次。adjust: 'qfq' 或 ''（raw）。"""
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_daily(symbol=to_sym(code), start_date=START,
                                     end_date=END, adjust=adjust)
            if df is not None and not df.empty:
                out = df.rename(columns={"date": "date", "open": "open",
                                         "close": "close", "high": "high",
                                         "low": "low", "volume": "volume",
                                         "amount": "amount"})
                for c in ["turnover"]:
                    if c not in out.columns:
                        out[c] = float("nan")
                out["code"] = code
                out["date"] = pd.to_datetime(out["date"])
                return out[["date", "open", "high", "low", "close", "volume",
                            "amount", "turnover", "code"]]
        except Exception as e:
            print(f"    [retry {attempt+1}/3 {code} {adjust or 'raw'}] {type(e).__name__}")
        time.sleep(2 * (attempt + 1))
    return None


def work(code: str) -> tuple:
    qfq_path = os.path.join(OUT, "qfq", f"{code}.parquet")
    raw_path = os.path.join(OUT, "raw", f"{code}.parquet")
    if os.path.exists(qfq_path) and os.path.exists(raw_path):
        return code, "skip"
    n = 0
    ok_q = os.path.exists(qfq_path)
    if not ok_q:
        df = fetch_one(code, "qfq")
        # 行数门槛 800：次新股 IPO 晚合法短历史（如 301029=1232 行）不应被拒；
        # 短历史股票由协整筛选的 IS≥500 检查自然淘汰
        if df is not None and len(df) > 800 and (df["close"] > 0).all():
            df.to_parquet(qfq_path, index=False)
            ok_q = True
            n = len(df)
    ok_r = os.path.exists(raw_path)
    if not ok_r:
        df = fetch_one(code, "")
        if df is not None and len(df) > 800 and (df["close"] > 0).all():
            df.to_parquet(raw_path, index=False)
            ok_r = True
            n = max(n, len(df))
    return code, "ok" if (ok_q and ok_r) else "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=6)
    ap.add_argument("--chunk-id", type=int, default=0)
    args = ap.parse_args()

    cache = os.path.join(PROJ2, "data", "pairs", "universe_csi1000.csv")
    codes = sorted(pd.read_csv(cache, encoding="utf-8-sig")["code"].astype(str).str.zfill(6))
    chunk = codes[args.chunk_id::args.n_chunks]
    for adj in ("qfq", "raw"):
        os.makedirs(os.path.join(OUT, adj), exist_ok=True)
    print(f"[chunk {args.chunk_id}/{args.n_chunks}] {len(chunk)} 只 | 新浪源 | 断点续传")

    t0 = time.time()
    done = fail = skip = 0
    for i, code in enumerate(chunk, 1):
        _, st = work(code)
        if st == "skip":
            skip += 1
        elif st == "ok":
            done += 1
        else:
            fail += 1
        if i % 50 == 0 or i == len(chunk):
            print(f"  [chunk{args.chunk_id}] {i}/{len(chunk)} done={done} fail={fail} "
                  f"skip={skip} elapsed={(time.time()-t0)/60:.1f}min")
        time.sleep(0.2)   # 限速
    print(f"[chunk {args.chunk_id} DONE] done={done} fail={fail} skip={skip} "
          f"| {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()

