r"""缓存扩展池代码清单（CSI1000 - hs800），供快取脚本离线读取。"""
import os
import time

import akshare as ak
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE = os.path.join(PROJ2, "data", "pairs", "universe_csi1000.csv")


def main() -> None:
    uni = pd.read_parquet(r"F:\Deepseekwork\秋招\project1_factor\data\raw\universe.parquet")
    hs800 = set(uni["code"].astype(str).str.zfill(6))
    for attempt in range(5):
        try:
            c1000 = ak.index_stock_cons_csindex(symbol="000852")
            codes = sorted(set(c1000["成分券代码"].astype(str).str.zfill(6)) - hs800)
            pd.DataFrame({"code": codes}).to_csv(CACHE, index=False, encoding="utf-8-sig")
            print(f"OK: {len(codes)} 只 -> {CACHE}")
            return
        except Exception as e:
            print(f"[retry {attempt+1}/5] {type(e).__name__}: {str(e)[:80]}")
            time.sleep(5 * (attempt + 1))
    raise SystemExit("csindex 接口持续失败")


if __name__ == "__main__":
    main()
