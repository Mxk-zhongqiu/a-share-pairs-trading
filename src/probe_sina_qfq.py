r"""探测新浪 qfq 对疑似复权 bug 股票是否干净（p2 修复决策用）。

判定: qfq 单日收益 |ret| 应不超过涨跌停上限（主板 10%/创业板科创板 20%）；
      |ret|>21% 的日数 = 0 才认为该源干净（对 >21% 的异常类问题）。
"""
import akshare as ak
import pandas as pd

CHECKS = ["600066", "000983", "601899", "000858", "002714", "601717",
          "603129", "601001", "000830", "600348", "605117", "600132"]


def to_sym(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def main() -> None:
    for code in CHECKS:
        try:
            df = ak.stock_zh_a_daily(symbol=to_sym(code), start_date="20170101",
                                     end_date="20260821", adjust="qfq")
            ret = df["close"].pct_change()
            n_ext = int((ret.abs() > 0.21).sum())
            last = df["date"].iloc[-1]
            n_rows = len(df)
            print(f"{code}: sina qfq rows={n_rows} ext>21%={n_ext} last={last}")
        except Exception as e:
            print(f"{code}: FAIL {type(e).__name__} {str(e)[:80]}")


if __name__ == "__main__":
    main()
