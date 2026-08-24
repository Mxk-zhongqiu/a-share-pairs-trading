r"""卡尔曼时变 β 估计（p3b 主模型，项目二控制主场）。

口径（预注册，见 docs/OU_KALMAN.md，防未来函数硬约束逐条落实）:
    状态: x_t = [α_t, β_t]ᵀ, 随机游走 x_{t+1} = x_t + η, η~N(0,Q)
    观测: la_t = α_t + β_t·lb_t + ε_t,  ε~N(0,R),  H_t = [1, lb_t]
    标定: R = IS EG 残差方差;  Q = R·diag(λ_α, λ_β), λ 网格在 IS 内滚动预测误差选优
    因果性: 前向滤波（无 smoother）; 参数标定只用 IS; 交易期冻结 Q/R
    输出: 每对 filtered 状态 + 预测残差 e_t（可交易价差）+ 滚动 z-score

用法:
    & python kalman.py --max-pairs 20          # 冒烟
    & python kalman.py                          # 全量
输出: data/signals/kalman_<pair>.parquet（date, beta, alpha, spread_e, z60）
      data/pairs/kalman_summary.csv（每对 Q/R 选择与 IS 预测误差）
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

PROJ2 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ALIGNED = os.path.join(PROJ2, "data", "aligned")
PAIRS = os.path.join(PROJ2, "data", "pairs")
SIGNALS = os.path.join(PROJ2, "data", "signals")
IS_END = "2021-12-31"
Z_WINDOW = 60                 # 预注册：z-score 滚动窗口（交易日）


def kalman_filter(la: np.ndarray, lb: np.ndarray, R: float,
                  Q: np.ndarray, x0: np.ndarray, P0: np.ndarray) -> dict:
    """前向卡尔曼滤波。返回 filtered 状态/协方差与预测残差（因果）。"""
    n = len(la)
    x = x0.copy()
    P = P0.copy()
    betas = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    resid = np.full(n, np.nan)      # 预测残差 e_t = la - H x_{t|t-1}（t 收盘已知）
    for t in range(n):
        H = np.array([1.0, lb[t]])
        # predict
        x_pred = x
        P_pred = P + Q
        # innovation
        e = la[t] - H @ x_pred
        S = H @ P_pred @ H + R
        # update
        K = P_pred @ H / S
        x = x_pred + K * e
        P = (np.eye(2) - np.outer(K, H)) @ P_pred
        betas[t] = x[1]
        alphas[t] = x[0]
        resid[t] = e
    return {"beta": betas, "alpha": alphas, "resid": resid}


def rolling_z(s: np.ndarray, window: int) -> np.ndarray:
    """滚动 z-score（只用 t 及以前窗口，无前视）。"""
    z = np.full(len(s), np.nan)
    for t in range(window, len(s)):
        w = s[t - window:t]
        sd = w.std(ddof=0)
        z[t] = (s[t] - w.mean()) / sd if sd > 1e-12 else np.nan
    return z


def select_lambda(la: np.ndarray, lb: np.ndarray, R: float,
                  grid: list, warmup: int = 120) -> tuple:
    """IS 内滚动预测误差选 λ（Q = R·diag(λ,λ)）。返回 (λ*, mse)。"""
    best, best_mse = grid[0], np.inf
    for lam in grid:
        Q = R * lam * np.eye(2)
        out = kalman_filter(la, lb, R, Q, np.array([0.0, 1.0]), 10.0 * R * np.eye(2))
        e = out["resid"][warmup:]
        mse = float(np.mean(e ** 2)) if len(e) else np.inf
        if mse < best_mse:
            best, best_mse = lam, mse
    return best, best_mse


def process_pair(pr: str, r: pd.Series, aligned_dir: str) -> pd.DataFrame | None:
    m = pd.read_parquet(os.path.join(aligned_dir, f"{pr}.parquet"))
    m = m.dropna(subset=["close_a", "close_b"])
    la = np.log(m["close_a"].to_numpy(float))
    lb = np.log(m["close_b"].to_numpy(float))
    dates = pd.to_datetime(m["date"]).to_numpy()
    is_mask = dates <= np.datetime64(IS_END)

    # 标定（只用 IS）
    la_is, lb_is = la[is_mask], lb[is_mask]
    R = float(np.var(la_is - (r["beta_is"] * lb_is + r["alpha_is"]), ddof=1))
    lam, mse = select_lambda(la_is, lb_is, R, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    # 前向滤波（全样本，因果：t 只用 ≤t）
    Q = R * lam * np.eye(2)
    x0 = np.array([r["alpha_is"], r["beta_is"]])
    P0 = 10.0 * R * np.eye(2)
    out = kalman_filter(la, lb, R, Q, x0, P0)

    z60 = rolling_z(out["resid"], Z_WINDOW)
    df = pd.DataFrame({
        "date": pd.to_datetime(dates), "beta": out["beta"], "alpha": out["alpha"],
        "spread_e": out["resid"], "z60": z60,
        "la": la, "lb": lb,       # log 价格（关系破裂检验用冻结 β 价差）
    })
    return df, {"pair": pr, "R": R, "lambda": lam, "mse_is": mse,
                "beta0": r["beta_is"], "alpha0": r["alpha_is"]}


def main() -> None:
    import sys as _s
    ap = None
    aligned_dir = _s.argv[1] if len(_s.argv) > 1 else ALIGNED
    sel = pd.read_csv(os.path.join(PAIRS, "coint_selected.csv"), encoding="utf-8-sig")
    if "--max" in _s.argv:
        sel = sel.head(int(_s.argv[_s.argv.index("--max") + 1]))
    print(f"[pairs] 稳定协整对 {len(sel)} | aligned={aligned_dir}")

    os.makedirs(SIGNALS, exist_ok=True)
    t0 = time.time()
    summaries = []
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        pr = r["pair"]
        try:
            df, summ = process_pair(pr, r, aligned_dir)
            df.to_parquet(os.path.join(SIGNALS, f"kalman_{pr}.parquet"), index=False)
            summaries.append(summ)
        except Exception as e:
            print(f"  [ERR] {pr}: {type(e).__name__} {str(e)[:80]}")
        if i % 10 == 0 or i == len(sel):
            print(f"[{i}/{len(sel)}] elapsed={(time.time()-t0)/60:.1f}min")

    summ_df = pd.DataFrame(summaries)
    summ_df.to_csv(os.path.join(PAIRS, "kalman_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"DONE -> data/signals/kalman_*.parquet ({len(summ_df)} 对)  ({time.time()-t0:.0f}s)")
    if not summ_df.empty:
        print(summ_df.to_string(index=False))


if __name__ == "__main__":
    main()
