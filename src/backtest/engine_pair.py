"""配对账本引擎 v4：股数记账 + 现金预算硬约束（无杠杆）+ 多空双腿 + 融券费。

与项目一 engine.py 的关系（交接文档 §1"引擎零改造"的诚实落地）:
    复用的机制: costs.py（佣金/印花税日期分段/滑点）、constraints.py（涨跌停/停牌）、
               metrics.py（指标）、股数记账+现金预算无杠杆纪律（P4）。
    新增能力: 支持负股数（空头腿）与融券费——配对交易是市场中性多空结构，
             项目一引擎是纯多头，空头支持是项目二策略层的必要扩展（本文件）。

口径:
    - 信号日 T 收盘 → 目标仓位按调仓日 D 生效（D 收盘成交，与项目一一贯）
    - 状态: cash + {code: shares}(可为负); NAV = cash + Σ shares×close(ffill)
    - 调仓按 per-stock 目标权重（策略层已把"对"翻译成两只股票的权重）
    - delta<0 = 卖出方向（平多 / 加空）→ 现金流入（受 sellable 约束: 非跌停）
    - delta>0 = 买入方向（加多 / 平空）→ 现金流出（受 buyable 约束: 非涨停 + 现金预算）
    - 现金预算硬约束: 买入总额 ≤ 现金/(1+费率) → 现金永不为负 → 无杠杆
    - 融券费: 空头市值 × rate/252 每日（预注册 8%/年，报告披露假设）
    - 停牌/无行情: 持仓冻结（价格 ffill），超过 60 日强制清算（兜底）
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .constraints import tradability_at
from .costs import CostConfig, trade_costs

STALE_LIQUIDATE_DAYS = 60
SHORT_FEE_RATE = 0.08        # 融券费率 8%/年（假设，报告披露）


class PairBookEngine:
    def __init__(
        self,
        close_df: pd.DataFrame,
        limit_up_df: pd.DataFrame,
        limit_down_df: pd.DataFrame,
        cost_cfg: CostConfig | None = None,
        short_fee_rate: float = SHORT_FEE_RATE,
    ):
        """close/limit 均为 index=date, columns=code 矩阵。"""
        self.close = close_df
        self.close_ff = close_df.ffill()
        self.limit_up = limit_up_df
        self.limit_down = limit_down_df
        self.cfg = cost_cfg or CostConfig()
        self.dates = close_df.index
        self.short_fee_rate = short_fee_rate
        self.last_w: Dict[str, float] = {}   # code -> 上次目标权重（未变持仓不重估）

    def run(self, targets: List[Tuple[pd.Timestamp, Dict[str, float]]],
            exec_log: List[dict] | None = None):
        """targets: [(调仓日, {code: 目标权重})]。权重可为负（空头），Σ|w|≤1 由策略层保证。
        exec_log: 可选，逐笔执行记录（date/code/side/amount/fee），诊断用。"""
        rb_map = {pd.Timestamp(d): w for d, w in targets}
        nav = pd.Series(np.nan, index=self.dates)
        shares: Dict[str, float] = {}
        cash = 1.0
        last_valid: Dict[str, int] = {}
        turnover_series = pd.Series(dtype=float)
        costs_series = pd.Series(dtype=float)
        short_fee_total = 0.0

        for i, d in enumerate(self.dates):
            close_row = self.close.loc[d]
            ff_row = self.close_ff.loc[d]
            for code in list(shares):
                if pd.notna(close_row.get(code)):
                    last_valid[code] = i

            # 每日融券费（空头市值 × rate/252）
            short_mv = sum(sh * ff_row.get(c, 0.0)
                           for c, sh in shares.items() if sh < 0)
            if short_mv < 0:
                fee = -short_mv * self.short_fee_rate / 252.0
                cash -= fee
                short_fee_total += fee

            if d in rb_map:
                shares, cash, cost, sell_amt, buy_amt, execs = self._rebalance(
                    d, i, rb_map[d], shares, cash, last_valid)
                turnover_series.loc[d] = (sell_amt + buy_amt) / 2.0
                costs_series.loc[d] = cost
                if exec_log is not None:
                    exec_log.extend(execs)

            mv = cash + sum(sh * ff_row.get(c, 0.0) for c, sh in shares.items())
            nav.loc[d] = mv

        return nav, turnover_series, costs_series, short_fee_total

    # ── 内部 ──────────────────────────────────────────────
    def _rebalance(self, d, di, target_w, shares, cash, last_valid):
        """收盘调仓：只交易目标权重变化的代码（未变持仓固定股数漂移，避免反复重配平）。

        语义: target_w 为全书目标权重（可为负）；code 未出现 = 目标 0（平仓）。
              与上次目标权重相同的代码跳过（保持股数）——多对重叠股票由策略层
              per-code 权重加总表达，这里按 code 增量执行。
        """
        trad = tradability_at(self.close.loc[d], self.limit_up.loc[d],
                              self.limit_down.loc[d])
        close_row = self.close.loc[d]
        ff_row = self.close_ff.loc[d]

        cur_amount: Dict[str, float] = {}
        for code, sh in shares.items():
            px = close_row.get(code)
            if pd.isna(px):
                px = ff_row.get(code, np.nan)
            cur_amount[code] = sh * px if pd.notna(px) else 0.0

        total_mv = sum(cur_amount.values()) + cash
        tgt_amount = {c: w * total_mv for c, w in target_w.items()}

        # 只处理"目标权重有变化"的代码 + stale 强平兜底（数据截断/长期停牌）
        touched = []
        for code in set(tgt_amount) | set(self.last_w):
            w_new = tgt_amount.get(code, 0.0)
            w_last = self.last_w.get(code, 0.0)
            if abs(w_new - w_last) < 1e-9:
                continue
            touched.append(code)
        for code in shares:
            if (di - last_valid.get(code, di)) > STALE_LIQUIDATE_DAYS:
                touched.append(code)   # 无论目标是否变化都强制处理
        touched = list(dict.fromkeys(touched))

        # 融券款冻结: 空头市值作为保证金不可再用于买入（A 股实盘口径）
        frozen = -sum(min(0.0, sh * (ff_row.get(c, 0.0) or 0.0))
                      for c, sh in shares.items())

        new_shares = dict(shares)
        new_cash = cash
        sell_amt = buy_amt = fee_total = 0.0
        achieved: set = set()          # 目标权重实际达成的代码（未达成的留待下次重试）
        execs: List[dict] = []

        # 1) 卖出方向（平多 / 加空 / 开新空 / stale 强平多头）
        for code in touched:
            amt = cur_amount.get(code, 0.0)
            t = tgt_amount.get(code, 0.0)
            stale = (di - last_valid.get(code, di)) > STALE_LIQUIDATE_DAYS
            if stale and amt > 0:
                delta = -amt            # stale 多头: 强制全部卖出
            else:
                delta = t - amt
            if delta > -1e-9:
                continue
            px = close_row.get(code)
            if pd.isna(px):
                px = ff_row.get(code, np.nan)
            sellable = trad.get(code, {}).get("sellable", False) or stale
            if sellable and pd.notna(px) and px > 0:
                sell_amt += -delta
                fee = trade_costs(0.0, -delta, self.cfg, d)
                new_shares[code] = new_shares.get(code, 0.0) - (-delta) / px
                new_cash += (-delta) - fee
                fee_total += fee
                achieved.add(code)
                execs.append({"date": d, "code": code, "side": "sell",
                              "amount": -delta, "fee": fee, "px": px,
                              "blocked": False})
                if abs(new_shares[code]) < 1e-9:
                    del new_shares[code]
            else:
                execs.append({"date": d, "code": code, "side": "sell",
                              "amount": -delta, "fee": 0.0, "px": np.nan,
                              "blocked": True})

        # 2) 买入方向（加多 / 平空 / stale 强平空头）: 现金预算 = 现金 − 冻结融券款
        buy_rate = self.cfg.commission_rate + self.cfg.slippage
        for code in touched:
            cur = cur_amount.get(code, 0.0)
            t = tgt_amount.get(code, 0.0)
            stale = (di - last_valid.get(code, di)) > STALE_LIQUIDATE_DAYS
            if stale and cur < 0:
                delta = -cur            # stale 空头: 强制全部买回
            else:
                delta = t - cur
            if delta > 1e-9 and (trad.get(code, {}).get("buyable", False) or stale):
                px = close_row.get(code)
                if pd.isna(px):
                    px = ff_row.get(code, np.nan)
                if pd.notna(px) and px > 0:
                    budget = max(0.0, new_cash - frozen)
                    buy_amount = min(delta, budget / (1.0 + buy_rate))
                    if buy_amount > 1e-9:
                        fee = trade_costs(buy_amount, 0.0, self.cfg, d)
                        new_shares[code] = new_shares.get(code, 0.0) + buy_amount / px
                        new_cash -= buy_amount + fee
                        buy_amt += buy_amount
                        fee_total += fee
                        achieved.add(code)
                        execs.append({"date": d, "code": code, "side": "buy",
                                      "amount": buy_amount, "fee": fee, "px": px,
                                      "blocked": False})
                    else:
                        execs.append({"date": d, "code": code, "side": "buy",
                                      "amount": 0.0, "fee": 0.0, "px": np.nan,
                                      "blocked": True})
                else:
                    execs.append({"date": d, "code": code, "side": "buy",
                                  "amount": 0.0, "fee": 0.0, "px": np.nan,
                                  "blocked": True})

        # last_w 只记录实际达成的目标；受阻代码保留旧值 → 下次调仓自动重试
        new_last = dict(self.last_w)
        for code in achieved:
            w_tgt = target_w.get(code, 0.0)
            if abs(w_tgt) > 1e-9:
                new_last[code] = w_tgt
            else:
                new_last.pop(code, None)
        self.last_w = new_last
        return new_shares, new_cash, fee_total, sell_amt, buy_amt, execs
