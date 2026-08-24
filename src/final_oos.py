r"""纯样本外回测（P0-4）：终版参数冻结 → 2024-2026 真样本外 + bootstrap 置信区间。

协议（预注册 docs/UPGRADE.md）：参数在 2022-2023 选定后**冻结**，2024-2026 零调整、
零重选、零事后确认。bootstrap：月度收益重采样 1000 次 → 夏普/年化 95% CI。
"""
import os
import sys

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJ2)
from backtest.costs import CostConfig
from backtest.engine_pair import PairBookEngine
from backtest.metrics import compute_metrics
from strategy.pair_strategy import build_targets

SIGNALS = os.path.join(PROJ2, "data", "signals")
BT = os.path.join(PROJ2, "data", "backtest_oos")   # 严格书（协议 v3）
REPORTS = os.path.join(PROJ2, "reports")
OOS_START = "2024-01-01"    # 纯样本外交易起点（参数冻结）


def bootstrap_ci(monthly_ret: pd.Series, n=1000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    r = monthly_ret.to_numpy(float)
    anns, shs = [], []
    for _ in range(n):
        b = rng.choice(r, size=len(r), replace=True)
        m = b.mean()
        sd = b.std(ddof=1)
        anns.append((1 + m) ** 12 - 1)
        shs.append(m * np.sqrt(12) / sd if sd > 0 else np.nan)
    anns = np.array(anns); shs = np.array(shs)
    return {"annual_ci": (np.nanpercentile(anns, 2.5), np.nanpercentile(anns, 97.5)),
            "sharpe_ci": (np.nanpercentile(shs, 2.5), np.nanpercentile(shs, 97.5)),
            "p_sharpe_gt0": float(np.nanmean(shs > 0))}


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

    # 终版参数（从 sensitivity_sel2023.csv 选定，此处作为参数传入；冻结）
    ap = sys.argv
    kw = dict(z_entry=float(ap[1]) if len(ap) > 1 else 1.5,
              z_exit=float(ap[2]) if len(ap) > 2 else 0.0,
              z_stop=float(ap[3]) if len(ap) > 3 else 3.0,
              w0=float(ap[4]) if len(ap) > 4 else 0.3,
              max_active=3,
              trade_start=OOS_START)   # 2024 冷启动：纯样本外从净值 1.0 起算

    targets, trades = build_targets(signals, **kw)
    eng = PairBookEngine(close_df, up, down, CostConfig(), short_fee_rate=0.08)
    exec_log = []
    nav, turn, costs, sfee = eng.run(targets, exec_log=exec_log)
    nav_oos = nav[nav.index >= OOS_START]
    m = compute_metrics(nav_oos, turn)

    print(f"[strategy] 参数 {kw} | 交易 {len(trades)} 事件 | 交易期 {len(nav_oos)} 日")
    print(f"[OOS 2024-2026] 净值={nav_oos.iloc[-1]:.4f} 总收益={m.get('total_ret'):+.2%} "
          f"年化={m.get('annual_ret'):+.2%} 夏普={m.get('sharpe'):+.2f} "
          f"回撤={m.get('max_drawdown'):.1%}")
    if m.get("monthly") is not None and len(m["monthly"]) > 5:
        ci = bootstrap_ci(m["monthly"])
        print(f"[bootstrap] 年化 95%CI [{ci['annual_ci'][0]:+.1%}, {ci['annual_ci'][1]:+.1%}]")
        print(f"[bootstrap] 夏普 95%CI [{ci['sharpe_ci'][0]:+.2f}, {ci['sharpe_ci'][1]:+.2f}] "
              f"| P(夏普>0)={ci['p_sharpe_gt0']:.1%}")

    # 保存
    nav.to_csv(os.path.join(REPORTS, "final_oos_nav.csv"))
    trades.to_csv(os.path.join(REPORTS, "final_oos_trades.csv"), index=False, encoding="utf-8-sig")
    if exec_log:
        pd.DataFrame(exec_log).to_csv(os.path.join(REPORTS, "final_oos_exec.csv"),
                                      index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
