r"""3 对核心终版配置 + 融券费敏感性（预注册 {0,4,8,10}）。"""
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
BT = os.path.join(PROJ2, "data", "backtest")
REPORTS = os.path.join(PROJ2, "reports")
CORE3 = ["600369_601881", "600098_600863", "600369_601555"]

CFG = dict(z_entry=2.5, z_exit=0.5, z_stop=3.5, max_hold=60, w0=0.5, max_active=3)


def main() -> None:
    close_df = pd.read_parquet(os.path.join(BT, "close_matrix.parquet"))
    up = pd.read_parquet(os.path.join(BT, "limit_up_matrix.parquet"))
    down = pd.read_parquet(os.path.join(BT, "limit_down_matrix.parquet"))
    signals = {pr: pd.read_parquet(os.path.join(SIGNALS, f"kalman_{pr}.parquet"))
               for pr in CORE3}

    targets, trades = build_targets(signals, **CFG)
    print(f"[strategy] {len(trades)} 事件")
    print(f"\n== 融券费敏感性（z_entry=2.5/exit=0.5/stop=3.5/w0=0.5）==")
    for fee in [0.0, 0.04, 0.08, 0.10]:
        eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=fee)
        nav, turn, costs, sfee = eng.run(targets)
        m = compute_metrics(nav[nav.index >= pd.Timestamp("2022-01-01")], turn)
        print(f"  融券费 {fee:.0%}: 净值={nav.iloc[-1]:.4f} 年化={m.get('annual_ret'):+.2%} "
              f"夏普={m.get('sharpe'):+.2f} 回撤={m.get('max_drawdown'):.1%} 融券费支出={sfee:.3f}")

    # 终版（8%）保存全套输出
    eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=0.08)
    exec_log = []
    nav, turn, costs, sfee = eng.run(targets, exec_log=exec_log)
    m = compute_metrics(nav[nav.index >= pd.Timestamp("2022-01-01")], turn)
    nav.to_csv(os.path.join(REPORTS, "final_nav.csv"))
    trades.to_csv(os.path.join(REPORTS, "final_trades.csv"), index=False, encoding="utf-8-sig")
    if exec_log:
        pd.DataFrame(exec_log).to_csv(os.path.join(REPORTS, "final_exec.csv"),
                                      index=False, encoding="utf-8-sig")
    m.pop("monthly", None)
    pd.Series(m).to_csv(os.path.join(REPORTS, "final_metrics.csv"), encoding="utf-8-sig")
    print(f"\n[final] 净值={nav.iloc[-1]:.4f} 年化={m.get('annual_ret'):+.2%} "
          f"夏普={m.get('sharpe'):+.2f} 回撤={m.get('max_drawdown'):.1%} "
          f"换手(调仓日均值)={m.get('avg_turnover'):.3f} 融券费={sfee:.3f} 交易={len(trades)}")


if __name__ == "__main__":
    main()
