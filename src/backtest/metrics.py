"""绩效指标：年化收益 / 波动 / 夏普 / 最大回撤 / 月度收益 / 换手率。"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(nav: pd.Series, turnover_series: pd.Series | None = None,
                    rf: float = 0.0) -> dict:
    """输入: nav 为组合净值序列（含起始日），turnover_series 为调仓期单边换手。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    rets = nav.pct_change().dropna()
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1.0
    annual_ret = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1.0 if n_years > 0 else np.nan
    ann_vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(rets) > 1 else np.nan
    sharpe = (rets.mean() * TRADING_DAYS - rf) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1.0).min())
    # 月度收益
    monthly = nav.resample("ME").last().pct_change().dropna()
    return {
        "total_ret": total_ret,
        "annual_ret": annual_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_days": len(rets),
        "avg_turnover": float(turnover_series.mean()) if turnover_series is not None and len(turnover_series) else np.nan,
        "monthly_ret_mean": float(monthly.mean()) if len(monthly) else np.nan,
        "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "monthly": monthly,
    }
