r"""修复 hs800 中 30 只股票的 qfq 复权 bug（项目二本地副本，不动项目一数据）。

背景（2026-08-23 发现，SOP 素材）:
    项目一 QA 抽检"非除权日"复权一致性，漏掉了只在除权调整日出现的因子错误
    —— 30 只高分红/送转股 qfq 单日收益出现 ±22%~37% 假跳变（raw 同日均在涨跌停内）。
    此类错误会直接污染配对交易的价差序列（假跳变 = 假协整/假信号），必须修复。
    探测确认新浪 qfq 数据干净（12/12 抽查 0 极端日）。

修复: 新浪 qfq 价格 + 腾讯 raw 量额换手（同 fix_qfq_sina.py 思路），
     验证门槛: 无零价 / 无 |ret|>21% 极端日 / 行数>1800 / 末日对齐。
输出: data/raw_fixed/qfq/<code>.parquet（项目二读取优先，见 qa_alignment --overlay）
"""
import os
import time

import akshare as ak
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(PROJ2, "data", "raw_fixed", "qfq")
P1_RAW = r"F:\Deepseekwork\秋招\project1_factor\data\raw"

# 30 只 qfq 极端日>3 的股票（2026-08-23 程序化检出，含高分红煤炭/银行/白酒/家电等）
AFFECTED = ["600066", "002714", "601717", "603129", "000983", "601001", "000830",
            "600348", "605117", "600132", "600873", "300100", "300394", "600329",
            "601216", "601838", "300972", "601899", "000858", "000921", "002318",
            "302132", "603345", "000568", "300570", "600346", "600350", "600985",
            "688036", "688676"]
START, END = "20170101", "20260821"
EXTREME_LIMIT = 0.21      # 超过涨跌停上限的收益视为异常（主板10%/双创20%，除复牌/新股日）


def to_sym(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def fetch_sina_qfq(code: str) -> pd.DataFrame | None:
    for attempt in range(4):
        try:
            df = ak.stock_zh_a_daily(symbol=to_sym(code), start_date=START,
                                     end_date=END, adjust="qfq")
            if df is not None and not df.empty:
                out = df[["date", "open", "high", "low", "close"]].copy()
                out["date"] = pd.to_datetime(out["date"])
                return out
        except Exception as e:
            print(f"    [retry {attempt+1}/4] {type(e).__name__}: {str(e)[:80]}")
            time.sleep(3 * (attempt + 1))
    return None


def validate(qfq: pd.DataFrame, code: str, n_raw: int) -> tuple:
    """验证修复后序列: 无零价 / 无极端收益 / 行数充足 / 末日对齐。"""
    n_zero = int((qfq["close"] <= 0).sum())
    ret = qfq["close"].pct_change()
    n_ext = int((ret.abs() > EXTREME_LIMIT).sum())
    last = qfq["date"].iloc[-1]
    ok = (n_zero == 0 and n_ext == 0 and len(qfq) > 1800)
    return ok, (n_zero, n_ext, len(qfq), last)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fixed, failed, skipped = [], [], []
    t0 = time.time()
    for i, code in enumerate(AFFECTED, 1):
        out_path = os.path.join(OUT, f"{code}.parquet")
        raw_path = os.path.join(P1_RAW, "raw", f"{code}.parquet")
        if not os.path.exists(raw_path):
            skipped.append((code, "raw 缺失"))
            continue
        n_raw = len(pd.read_parquet(raw_path, columns=["date"]))

        prices = fetch_sina_qfq(code)
        if prices is None:
            failed.append((code, "新浪拉取失败"))
            continue
        raw = pd.read_parquet(raw_path)[["date", "volume", "amount", "turnover"]]
        raw["date"] = pd.to_datetime(raw["date"])
        merged = prices.merge(raw, on="date", how="inner")
        merged["code"] = code
        merged = merged[["date", "open", "high", "low", "close",
                         "volume", "amount", "turnover", "code"]]

        ok, info = validate(merged, code, n_raw)
        if ok:
            merged.to_parquet(out_path, index=False)
            fixed.append(code)
            print(f"[{i}/{len(AFFECTED)}] {code}: OK zero={info[0]} ext={info[1]} "
                  f"rows={info[2]} last={info[3].date()}")
        else:
            failed.append((code, f"验证不过 zero={info[0]} ext={info[1]} rows={info[2]}"))
            print(f"[{i}/{len(AFFECTED)}] {code}: 验证不过, 不写入")

    print(f"\nDONE ({time.time()-t0:.0f}s): 修复 {len(fixed)} / 失败 {len(failed)} / 跳过 {len(skipped)}")
    if failed:
        print("失败:", failed)
    print(f"输出 -> {OUT}")


if __name__ == "__main__":
    main()
