import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ------------------------------
# Paths (match your folder layout)
# ------------------------------
DATA_DIR = "data"
RESULT_DIR = "result"

STOCK_SELECTED_PATH = os.path.join(RESULT_DIR, "stock_selected.csv")
DAILY_PRICE_PATH = os.path.join(DATA_DIR, "sp500_tickers_daily_price.csv")
OUT_PATH = os.path.join(RESULT_DIR, "portfolio_weights.csv")

# ------------------------------
# Hyperparameters
# ------------------------------
LOOKBACK_DAYS = 252      # trailing window length for covariance
EPS_COV = 1e-6           # tiny ridge for numerical stability


# ------------------------------
# Utilities
# ------------------------------
def _detect_price_column(df: pd.DataFrame) -> str:
    """
    Detect a reasonable price column from the daily price file.
    Preference order: 'adj_close', 'adjcp', 'close', 'prccd', 'price'.
    """
    candidates = ["adj_close", "adjcp", "close", "prccd", "price"]
    lower_cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_cols:
            return lower_cols[name]
    raise ValueError(
        f"Cannot find a price column among {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_prices_to_wide(path: str) -> pd.DataFrame:
    """
    Load daily prices and pivot to wide format: index=date, columns=tic, values=price.
    Accepts 'date' or 'datadate' as the date column. Automatically detects price column.
    """
    df = pd.read_csv(path)
    if "date" not in df.columns and "datadate" in df.columns:
        df = df.rename(columns={"datadate": "date"})
    if "tic" not in df.columns:
        raise ValueError("Daily price file must contain a 'tic' column.")

    price_col = _detect_price_column(df)
    keep = df[["date", "tic", price_col]].dropna()
    keep["date"] = pd.to_datetime(keep["date"])
    wide = keep.pivot(index="date", columns="tic", values=price_col).sort_index()
    return wide


def trailing_cov(px_wide: pd.DataFrame, asof_date: pd.Timestamp,
                 tickers: list[str], lookback: int) -> np.ndarray:
    """
    Compute trailing covariance matrix using percentage returns over the lookback window.
    Uses fill_method=None to avoid the FutureWarning and then fills remaining NaNs with 0.
    """
    end = pd.to_datetime(asof_date)
    window = px_wide.loc[: end - pd.Timedelta(days=1)].tail(lookback)
    rets = window[tickers].pct_change(fill_method=None).dropna(how="all").fillna(0.0)

    if len(rets) < max(20, lookback // 6):
        # fallback to shorter window if sample is too small
        rets = px_wide.loc[: end - pd.Timedelta(days=1)].tail(60).pct_change(fill_method=None).fillna(0.0)

    if rets.shape[0] > 1:
        cov = np.cov(rets.values.T)
    else:
        # if only one row, fall back to diagonal with per-series variance
        cov = np.diag(np.var(rets.values, axis=0))

    cov = np.atleast_2d(cov) + np.eye(len(tickers)) * EPS_COV
    return cov


def neg_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Negative Sharpe ratio with rf=0."""
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
    return -ret / vol


def max_sharpe_long_only(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Long-only maximum Sharpe. If the optimizer fails, fall back to long-only minimum variance.
    """
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(neg_sharpe, x0=x0, args=(mu, cov), method="SLSQP",
                   bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    if res.success:
        w = np.clip(res.x, 0, 1)
        return w / w.sum()

    # Fallback: long-only minimum variance
    def var_obj(w): return float(w @ cov @ w)
    res2 = minimize(var_obj, x0=x0, method="SLSQP",
                    bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    w = res2.x if res2.success else x0
    w = np.clip(w, 0, 1)
    return w / w.sum()


def align_to_trading_day(target_date: pd.Timestamp, trading_index: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Align a calendar date to the next available trading day.
    If the target is beyond the last trading day, clamp to the last index element.
    """
    target_date = pd.to_datetime(target_date)
    idx = trading_index.get_indexer([target_date], method="backfill")[0]
    if idx == -1:
        return trading_index[-1]
    return trading_index[idx]


def expand_daily(weights_by_reb_date: dict,
                 rebalance_dates: list[pd.Timestamp],
                 trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Expand quarterly weights to daily weights until the day before the next rebalance.
    For the last period, expand until the last available trading day.
    """
    rows = []
    for i, d in enumerate(rebalance_dates):
        start = d
        end = trading_index[-1] if i == len(rebalance_dates) - 1 else rebalance_dates[i + 1] - pd.Timedelta(days=1)
        days = trading_index[(trading_index >= start) & (trading_index <= end)]
        wmap = weights_by_reb_date[d]  # dict[tic] -> weight
        for day in days:
            for tic, wt in wmap.items():
                rows.append((day.strftime("%Y-%m-%d"), tic, float(wt)))
    return pd.DataFrame(rows, columns=["trade_date", "tic", "weights"])


# ------------------------------
# Main
# ------------------------------
def main():
    # 1) Load selected stocks
    sel = pd.read_csv(STOCK_SELECTED_PATH)
    need = {"tic", "predicted_return", "trade_date"}
    if not need.issubset(sel.columns):
        raise ValueError(f"stock_selected.csv must contain columns: {need}")
    sel["trade_date"] = pd.to_datetime(sel["trade_date"])

    # 2) Load daily prices and trading calendar
    px = load_prices_to_wide(DAILY_PRICE_PATH)
    trading_days = px.index

    # 3) Prepare pairs (original quarterly date, aligned trading date)
    rebalance_orig = sorted(pd.unique(sel["trade_date"]))
    pairs = []
    for d in rebalance_orig:
        d_trade = align_to_trading_day(d, trading_days)
        pairs.append((pd.to_datetime(d), d_trade))

    weights_by_date: dict[pd.Timestamp, dict] = {}

    # 4) Optimize per quarter
    for d_orig, d_trade in pairs:
        # tickers & mu must be taken by the ORIGINAL quarterly date
        tickers = sel.loc[sel["trade_date"] == d_orig, "tic"].dropna().unique().tolist()
        # keep only tickers existing in price matrix
        tickers = [t for t in tickers if t in px.columns]
        if len(tickers) == 0:
            print(f"[SKIP {d_orig.date()} -> {d_trade.date()}] no tickers with price coverage.")
            continue

        mu_map = (sel.loc[sel["trade_date"] == d_orig, ["tic", "predicted_return"]]
                    .drop_duplicates().set_index("tic")["predicted_return"].to_dict())
        mu = np.array([float(mu_map[t]) for t in tickers], dtype=float)

        cov = trailing_cov(px, d_trade, tickers, LOOKBACK_DAYS)
        w = max_sharpe_long_only(mu, cov)

        # store by the ALIGNED trading date (this is the rebal day used in backtest)
        weights_by_date[d_trade] = dict(zip(tickers, w))
        print(f"[{d_orig.date()} -> {d_trade.date()}] n={len(tickers)} | mean(mu)={mu.mean():.4f}")

    if not weights_by_date:
        raise RuntimeError("No quarterly weights were produced. Check data coverage and dates.")

    # 5) Expand to daily weights using the aligned dates we actually have
    rebalance_aligned = sorted(weights_by_date.keys())
    out = expand_daily(weights_by_reb_date=weights_by_date,
                       rebalance_dates=rebalance_aligned,
                       trading_index=trading_days)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved daily weights -> {OUT_PATH}")
    print(out.head())


if __name__ == "__main__":
    main()
