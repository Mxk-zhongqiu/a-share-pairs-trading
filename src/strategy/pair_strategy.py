r"""配对策略层（p4a）：卡尔曼价差信号 → β-对冲多空目标持仓 + 风控。

口径（预注册，见 docs/STRATEGY.md，v3 修订）:
    信号: 卡尔曼预测残差 e_t 的滚动 60 日 z（只用 t 及以前，无未来函数）
    执行: **信号日 T 收盘 → 目标在下一交易日 T+1 收盘成交**（与项目一一致，防同日前视）
    方向: z > +entry → A 相对贵 → 空 A 多 B；z < -entry → 反向
    仓位: β-对冲（gross=w0/对）: 空/多 B 名义额 = |β̂|/(1+|β̂|)·w0, A = 1/(1+|β̂|)·w0
          β̂ = 入场时卡尔曼滤波值，截断 [0.25, 2.5]；n_active·w0 ≤ 1（无杠杆）
    退出: |z| ≤ z_exit（回归）| |z| ≥ z_stop（止损）| 持有 ≥ max_hold（超时）
    止损冷却: stop 后 stop_cool 个完整交易日不重进
    关系破裂风控: 每 recheck_every 日对活跃对跑滚动 250 日 ADF，
          **检验对象 = 冻结入场 (α̂,β̂) 的价差 la − α̂_entry − β̂_entry·lb**
          （不是卡尔曼 innovation——innovation 近似白噪声，ADF 永不触发，v3 修复）
          p > break_p → 平仓 + 暂停 break_pause 日
    输出: 仅在状态跳变日输出目标（持仓期固定股数，引擎不重估）
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

BETA_CLIP = (0.25, 2.5)


def _leg_weights(beta: float, w0: float) -> Tuple[float, float]:
    b = float(np.clip(abs(beta), *BETA_CLIP))
    return w0 / (1.0 + b), w0 * b / (1.0 + b)   # (wA 大小, wB 大小)


def build_targets(
    signals: Dict[str, pd.DataFrame],
    z_entry: float = 2.0, z_exit: float = 0.5, z_stop: float = 3.5,
    max_hold: int = 60, w0: float = 0.3, max_active: int = 3,
    break_p: float = 0.05, break_pause: int = 20,
    trade_start: str = "2022-01-01", recheck_every: int = 5,
    stop_cool: int = 10,
) -> Tuple[List[Tuple[pd.Timestamp, Dict[str, float]]], pd.DataFrame]:
    """返回 (targets, 交易明细表)。signals: {pair: DataFrame(date,beta,alpha,z60,spread_e,la,lb)}。"""
    all_dates = sorted(set().union(*[set(pd.to_datetime(s["date"])) for s in signals.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(trade_start)]
    dates = pd.DatetimeIndex(all_dates)

    sig = {}
    for pr, s in signals.items():
        s = s.copy()
        s["date"] = pd.to_datetime(s["date"])
        sig[pr] = s.set_index("date").sort_index()

    pos: Dict[str, int] = {pr: 0 for pr in signals}       # +1 空A多B / -1 多A空B / 0 空仓
    hold: Dict[str, int] = {pr: 0 for pr in signals}
    pause: Dict[str, int] = {pr: 0 for pr in signals}
    cool: Dict[str, int] = {pr: 0 for pr in signals}      # 止损冷却（剩余禁入日）
    entry_beta: Dict[str, float] = {}
    entry_alpha: Dict[str, float] = {}
    trades: List[dict] = []
    targets_out: List[Tuple[pd.Timestamp, Dict[str, float]]] = []
    last_tgt: Dict[str, float] = {}

    def emit(sig_day, tgt):
        """目标在信号日次一交易日生效（T+1）。"""
        nonlocal last_tgt
        nxt = dates[dates > sig_day]
        if len(nxt) == 0:
            return
        targets_out.append((nxt[0], tgt))
        last_tgt = tgt

    for i, d in enumerate(dates):
        # 1) 关系破裂滚动检验（每 recheck_every 日；重估 β 后测残差平稳性 = "关系还在吗"）
        #    v3 修复: 冻结入场 β 会因 β 漂移频繁假破裂（实测 144 次/478 事件），
        #    正确检验 = 滚动窗口重估 (α,β) → 残差 ADF（交接文档 §3"滚动检验"本意）
        if i % recheck_every == 0:
            for pr in pos:
                if pos[pr] == 0 or pause[pr] > 0:
                    continue
                sdf = sig[pr]
                window = sdf.loc[:d].tail(250)
                if len(window) < 200:
                    continue
                la_w = window["la"].to_numpy(float)
                lb_w = window["lb"].to_numpy(float)
                X = np.column_stack([np.ones(len(lb_w)), lb_w])
                b, *_ = np.linalg.lstsq(X, la_w, rcond=None)
                resid = la_w - X @ b
                p = adfuller(resid, autolag="AIC")[1]
                if p > break_p:
                    pos[pr], hold[pr], pause[pr] = 0, 0, break_pause
                    trades.append({"date": d, "pair": pr, "action": "close_break",
                                   "z": float(window["z60"].iloc[-1]),
                                   "reason": f"break p={p:.3f}"})

        # 2) 状态机（开/平仓跳变；已开仓对子不被挤占，满仓时新对子等待空位）
        active_count = sum(1 for v in pos.values() if v != 0)
        for pr in list(pos):
            if pause[pr] > 0:
                pause[pr] -= 1
                continue
            if cool[pr] > 0:
                cool[pr] -= 1
                continue
            if d not in sig[pr].index:
                continue
            row = sig[pr].loc[d]
            if pd.isna(row["z60"]):
                continue
            z = float(row["z60"])
            if pos[pr] == 0:
                if active_count >= max_active:
                    continue          # 满仓：等待空位，不硬挤
                if z >= z_entry:
                    pos[pr], hold[pr] = 1, 1
                    entry_beta[pr] = float(row["beta"])
                    entry_alpha[pr] = float(row["alpha"])
                    active_count += 1
                    trades.append({"date": d, "pair": pr, "action": "open",
                                   "z": z, "beta": entry_beta[pr]})
                elif z <= -z_entry:
                    pos[pr], hold[pr] = -1, 1
                    entry_beta[pr] = float(row["beta"])
                    entry_alpha[pr] = float(row["alpha"])
                    active_count += 1
                    trades.append({"date": d, "pair": pr, "action": "open",
                                   "z": z, "beta": entry_beta[pr]})
            else:
                hold[pr] += 1
                reason = None
                if abs(z) <= z_exit:
                    reason = "revert"
                elif abs(z) >= z_stop:
                    reason = "stop"
                elif hold[pr] > max_hold:      # 持有超过 max_hold 交易日（含开仓日）
                    reason = "timeout"
                if reason:
                    trades.append({"date": d, "pair": pr, "action": "close",
                                   "z": z, "reason": reason})
                    pos[pr], hold[pr] = 0, 0
                    if reason == "stop":
                        cool[pr] = stop_cool   # 之后 stop_cool 个交易日禁入
                    entry_beta.pop(pr, None)
                    entry_alpha.pop(pr, None)

        # 3) 当日目标（全部在仓对子；仅在状态跳变日输出，T+1 生效）
        tgt: Dict[str, float] = {}
        for pr, pv in pos.items():
            if pv == 0:
                continue
            wA, wB = _leg_weights(entry_beta[pr], w0)
            ca, cb = pr.split("_")
            if pv == 1:                 # 空 A 多 B
                tgt[ca] = tgt.get(ca, 0.0) - wA
                tgt[cb] = tgt.get(cb, 0.0) + wB
            else:                       # 多 A 空 B
                tgt[ca] = tgt.get(ca, 0.0) + wA
                tgt[cb] = tgt.get(cb, 0.0) - wB
        if tgt != last_tgt:
            emit(d, tgt)

    return targets_out, pd.DataFrame(trades)
