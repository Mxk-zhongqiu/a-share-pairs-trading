r"""选参段扫描（P0-3）：参数网格在 2022-2023（选参段）评估，2024+ 不参与选参。

协议（预注册 docs/UPGRADE.md）：选参只用 2022-2023；选定后冻结，2024-2026 纯验证。
"""
import itertools
import os
import sys

import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJ2)
from backtest.costs import CostConfig
from backtest.engine_pair import PairBookEngine
from backtest.metrics import compute_metrics
from strategy.pair_strategy import build_targets

SIGNALS = os.path.join(PROJ2, "data", "signals")
BT = os.path.join(PROJ2, "data", "backtest_oos")   # 严格书（协议 v3）：1 对 600038_600765
REPORTS = os.path.join(PROJ2, "reports")
SEL_START, SEL_END = "2022-01-01", "2023-12-31"    # 选参段（2024+ 不参与选参）

GRID = dict(z_entry=[1.5, 2.0, 2.5], z_exit=[0.0, 0.5], z_stop=[3.0, 3.5, 4.0],
            w0=[0.3, 0.5])


def main() -> None:
    pairs = pd.read_csv(os.path.join(BT, "pair_list.csv"), encoding="utf-8-sig")
    close_df = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))
    up = pd.read_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down = pd.read_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    signals = {}
    for pr in pairs["pair"]:
        p = os.path.join(SIGNALS, f"kalman_{pr}.parquet")
        if os.path.exists(p):
            signals[pr] = pd.read_parquet(p)
    print(f"[pairs] {len(signals)} 对参与扫描 | 评估窗口 {SEL_START}~{SEL_END}")

    rows = []
    for combo in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, combo))
        targets, trades = build_targets(signals, **kw)
        eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=0.08)
        nav, turn, costs, sfee = eng.run(targets)
        m = compute_metrics(nav[(nav.index >= SEL_START) & (nav.index <= SEL_END)], turn)
        rows.append({**kw, "net_ret": m.get("total_ret", float("nan")),
                     "annual": m.get("annual_ret", float("nan")),
                     "sharpe": m.get("sharpe", float("nan")),
                     "dd": m.get("max_drawdown", float("nan")),
                     "n_trades": len(trades), "short_fee": sfee})

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out = os.path.join(REPORTS, "sensitivity_sel2023.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"DONE -> {out}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
