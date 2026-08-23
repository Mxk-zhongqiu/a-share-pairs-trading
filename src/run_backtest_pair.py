r"""配对交易回测运行器（p4b）：策略层 → 引擎 v4 → 指标 → 输出。

用法:
    & python run_backtest_pair.py                          # 默认参数
    & python run_backtest_pair.py --z-entry 2.0 --z-exit 0.5 --z-stop 3.5 \
        --max-hold 60 --w0 0.3 --max-active 3 --short-fee 0.08
输出 (reports/):
    pair_nav.csv / pair_trades.csv / pair_metrics.csv / pair_nav.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJ2)
from backtest.costs import CostConfig          # noqa: E402
from backtest.engine_pair import PairBookEngine  # noqa: E402
from backtest.metrics import compute_metrics    # noqa: E402
from strategy.pair_strategy import build_targets  # noqa: E402

SIGNALS = os.path.join(PROJ2, "data", "signals")
BT = os.path.join(PROJ2, "data", "backtest")
REPORTS = os.path.join(PROJ2, "reports")


def load_signals(pairs: pd.DataFrame) -> dict:
    out = {}
    for pr in pairs["pair"]:
        df = pd.read_parquet(os.path.join(SIGNALS, f"kalman_{pr}.parquet"))
        out[pr] = df
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--z-stop", type=float, default=3.5)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--w0", type=float, default=0.3)
    ap.add_argument("--max-active", type=int, default=3)
    ap.add_argument("--short-fee", type=float, default=0.08)
    ap.add_argument("--stop-cool", type=int, default=10)
    ap.add_argument("--trade-start", default="2022-01-01")
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    os.makedirs(REPORTS, exist_ok=True)
    close_df = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))
    up = pd.read_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down = pd.read_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    pairs = pd.read_csv(os.path.join(BT, "pair_list.csv"), encoding="utf-8-sig")

    signals = load_signals(pairs)
    targets, trades = build_targets(
        signals, z_entry=args.z_entry, z_exit=args.z_exit, z_stop=args.z_stop,
        max_hold=args.max_hold, w0=args.w0, max_active=args.max_active,
        stop_cool=args.stop_cool)
    print(f"[strategy] 交易明细 {len(trades)} 条")

    tag = args.tag
    engine = PairBookEngine(close_df, up, down, CostConfig(),
                            short_fee_rate=args.short_fee)
    exec_log: list = []
    nav, turnover, costs, short_fee_total = engine.run(targets, exec_log=exec_log)
    if exec_log:
        pd.DataFrame(exec_log).to_csv(os.path.join(REPORTS, f"pair_exec_{tag}.csv"),
                                      index=False, encoding="utf-8-sig")

    metrics = compute_metrics(nav[nav.index >= pd.Timestamp(args.trade_start)], turnover)
    metrics.update({"short_fee_total": short_fee_total,
                    "n_trades": len(trades),
                    "params": f"e{args.z_entry}/x{args.z_exit}/s{args.z_stop}/h{args.max_hold}/w{args.w0}/a{args.max_active}/f{args.short_fee}"})
    metrics.pop("monthly", None)      # 月度序列不入 csv（防类型污染）
    print(f"[engine] 期末净值 {nav.iloc[-1]:.4f} 年化 {metrics.get('annual_ret', float('nan')):.2%} "
          f"夏普 {metrics.get('sharpe', float('nan')):.2f} 回撤 {metrics.get('max_drawdown', float('nan')):.2%}")

    tag = args.tag
    nav.to_csv(os.path.join(REPORTS, f"pair_nav_{tag}.csv"))
    trades.to_csv(os.path.join(REPORTS, f"pair_trades_{tag}.csv"), index=False, encoding="utf-8-sig")
    pd.Series(metrics).to_csv(os.path.join(REPORTS, f"pair_metrics_{tag}.csv"), encoding="utf-8-sig")

    # 图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav.index, nav.values, label=f"Pair Book (sh {metrics.get('sharpe', float('nan')):.2f})")
    ax.set_title(f"Pairs Trading NAV [{tag}]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, f"pair_nav_{tag}.png"), dpi=110)
    plt.close(fig)
    print(f"DONE -> reports/pair_*_{tag}.csv/png")


if __name__ == "__main__":
    main()
