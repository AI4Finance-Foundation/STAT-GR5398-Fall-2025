# backtest_simple.py
# ------------------------------------------------------------
# Backtest a long-only quarterly (or daily) portfolio built from
#   - Prices:  data/sp500_tickers_daily_price.csv  (columns: date/datadate, tic, price_col)
#   - Weights: result/portfolio_weights.csv        (columns: trade_date, tic, weights)
# Optional benchmarks:
#   - data/SPX.csv, data/QQQ.csv (columns: date, close)
# Outputs (in --out_dir):
#   - backtest_pnl.csv (date, ret, nav)
#   - backtest_metrics.csv (ann_return, ann_vol, sharpe, cum_return, max_drawdown)
#   - backtest_equity.png (Portfolio vs (optional) SPX/QQQ)
# ------------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS_PER_YEAR = 252

# ----------------------- I/O helpers -----------------------
def detect_price_column(df: pd.DataFrame) -> str:
    """
    Auto-detect a usable price column.
    Preference order: adjusted close then close-like fields.
    """
    candidates = ["adj_close", "adjcp", "close", "prccd", "price"]
    lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower:
            return lower[k]
    raise ValueError(f"No price column among {candidates}. Got: {list(df.columns)}")

def load_prices_wide(path: str) -> pd.DataFrame:
    """
    Load long daily prices and pivot to wide format: index=date, columns=tic.
    If prccd & ajexdi both exist, use adjusted price = prccd / ajexdi.
    """
    df = pd.read_csv(path)
    if "date" not in df.columns and "datadate" in df.columns:
        df = df.rename(columns={"datadate": "date"})
    if "tic" not in df.columns:
        raise ValueError("Price file must contain column 'tic'.")

    if {"prccd", "ajexdi"}.issubset(df.columns):
        df["adj_close_auto"] = df["prccd"].astype(float) / df["ajexdi"].astype(float)
        price_col = "adj_close_auto"
    else:
        price_col = detect_price_column(df)

    keep = df[["date", "tic", price_col]].dropna()
    keep["date"] = pd.to_datetime(keep["date"])
    wide = keep.pivot(index="date", columns="tic", values=price_col).sort_index()
    wide = wide.replace([np.inf, -np.inf], np.nan)
    return wide


def load_weights_long(path: str) -> pd.DataFrame:
    """
    Load weights with columns ['trade_date','tic','weights'].
    Ensures long-only and renormalizes per trade_date if needed.
    """
    df = pd.read_csv(path)
    need = {"trade_date", "tic", "weights"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {need}")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["weights"] = df["weights"].astype(float)
    if (df["weights"] < 0).any():
        bad = df[df["weights"] < 0].head()
        raise ValueError(f"Negative weights encountered (no short sales). Sample:\n{bad}")

    def _norm(g):
        s = g["weights"].sum()
        if not np.isclose(s, 1.0, atol=1e-6) and s != 0:
            g["weights"] = g["weights"] / s
        return g
    df = df.groupby("trade_date", group_keys=False).apply(_norm)
    return df[["trade_date", "tic", "weights"]]

def load_index(path: str):
    """
    Load index series with columns [date, close]. Return Series indexed by date.
    If file not found, return None.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    s = df["close"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return s

# ----------------------- Backtest core -----------------------
def expand_or_align_weights_to_daily(weights_long: pd.DataFrame,
                                     price_index: pd.DatetimeIndex,
                                     t1: bool = False) -> pd.DataFrame:
    """
    Build daily weight matrix by forward-filling from rebalance dates.
    If t1=True, apply T+1 execution by shifting the whole daily matrix by 1 day.
    This version is robust to duplicated (trade_date, tic) rows.
    """
    # 1) de-duplicate per (trade_date, tic), keep last occurrence
    w = (weights_long.sort_values(["trade_date", "tic"])
                     .drop_duplicates(subset=["trade_date", "tic"], keep="last"))

    # 2) pivot to [rebalance_date x tic]; use pivot_table to be tolerant to duplicates
    Wq = (w.pivot_table(index="trade_date", columns="tic", values="weights", aggfunc="last")
            .sort_index())

    # 3) align to trading days and forward-fill
    Wd = Wq.reindex(price_index).ffill()

    # 4) T+1: shift the whole daily weight matrix by one trading day
    if t1:
        Wd = Wd.shift(1)

    # 5) fill NaNs before first rebalance with 0, then renormalize each day to sum=1
    Wd = Wd.fillna(0.0)
    s = Wd.sum(axis=1).replace(0, np.nan)
    Wd = Wd.div(s, axis=0).fillna(0.0)

    # 6) safety: ensure unique index (reindex requires unique labels)
    Wd = Wd[~Wd.index.duplicated(keep="last")]

    return Wd


def compute_portfolio_nav(prices_wide: pd.DataFrame,
                          daily_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily portfolio return and NAV using same-day weights.
    Returns a DataFrame with ['date','ret','nav'].
    """
    tickers = sorted(set(prices_wide.columns).intersection(set(daily_weights.columns)))
    if not tickers:
        raise ValueError("No overlapping tickers between price data and weights.")

    px = prices_wide[tickers].copy()
    W  = daily_weights[tickers].reindex(px.index).fillna(0.0)

    # Daily simple returns
    ret = px.pct_change(fill_method=None).fillna(0.0)

    # Portfolio daily return
    port_ret = (W * ret).sum(axis=1)

    nav = (1.0 + port_ret).cumprod()
    out = pd.DataFrame({"date": px.index, "ret": port_ret.values, "nav": nav.values})
    return out

# ----------------------- Metrics & plotting -----------------------
def perf_stats(pnl: pd.DataFrame) -> dict:
    """Compute key annualized metrics."""
    r = pnl["ret"].dropna()
    if r.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "cum_return": np.nan, "max_drawdown": np.nan}

    ann_ret = r.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe  = (ann_ret - 0.0) / ann_vol if ann_vol > 0 else np.nan

    eq = (1.0 + r).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    mdd = dd.min() if not dd.empty else np.nan

    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "cum_return": float(eq.iloc[-1] - 1.0),
        "max_drawdown": float(mdd),
    }

def plot_equity(port: pd.Series,
                spx: pd.Series | None,
                qqq: pd.Series | None,
                out_path: str,
                title: str = "Portfolio Values"):
    """
    Plot Portfolio value vs optional SPX/QQQ, normalized to 1 at start.
    """
    plt.figure(figsize=(9, 5))
    port.plot(label="Portfolio (with Quarterly Adjustment)", linewidth=1.8)
    if spx is not None and not spx.empty:
        spx_eq = (spx / spx.iloc[0])
        spx_eq.plot(label="SPX", linewidth=1.2)
    if qqq is not None and not qqq.empty:
        qqq_eq = (qqq / qqq.iloc[0])
        qqq_eq.plot(label="QQQ", linewidth=1.2)
    plt.title(title)
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser(description="Simple backtest for long-only quarterly portfolio (no short sales).")
    parser.add_argument("--price_csv", type=str, default="data/sp500_tickers_daily_price.csv",
                        help="Daily price file with columns [date/datadate, tic, price_col]")
    parser.add_argument("--weights_csv", type=str, default="result/portfolio_weights.csv",
                        help="Weights file with columns [trade_date, tic, weights]")
    parser.add_argument("--spx", type=str, default="data/SPX.csv", help="Optional SPX file [date,close]")
    parser.add_argument("--qqq", type=str, default="data/QQQ.csv", help="Optional QQQ file [date,close]")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--t1", action="store_true", help="Use T+1 execution (apply new weights from next trading day)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load prices (wide) and clip date range
    px = load_prices_wide(args.price_csv)
    if args.start:
        px = px.loc[px.index >= pd.to_datetime(args.start)]
    if args.end:
        px = px.loc[px.index <= pd.to_datetime(args.end)]

    # 2) Load weights (long) and expand to daily
    w = load_weights_long(args.weights_csv)
    Wd = expand_or_align_weights_to_daily(w, px.index, t1=args.t1)

    # 3) Compute portfolio PnL
    pnl = compute_portfolio_nav(px, Wd)
    pnl.to_csv(os.path.join(args.out_dir, "backtest_pnl.csv"), index=False)

    # 4) Metrics
    m = perf_stats(pnl)
    pd.DataFrame([m]).to_csv(os.path.join(args.out_dir, "backtest_metrics.csv"), index=False)
    print("=== Backtest Metrics ===")
    print(f"Annual Return : {m['ann_return']:.2%}")
    print(f"Annual Vol    : {m['ann_vol']:.2%}")
    print(f"Sharpe (rf=0) : {m['sharpe']:.3f}")
    print(f"Cumulative Ret: {m['cum_return']:.2%}")
    print(f"Max Drawdown  : {m['max_drawdown']:.2%}")

    # 5) Plot equity vs optional benchmarks
    spx = load_index(args.spx)
    qqq = load_index(args.qqq)

    # Align benchmarks to portfolio date range & normalize to start
    eq_port = pnl.set_index("date")["nav"]
    if spx is not None:
        spx = spx.reindex(eq_port.index).ffill()
    if qqq is not None:
        qqq = qqq.reindex(eq_port.index).ffill()

    plot_equity(eq_port, spx, qqq, os.path.join(args.out_dir, "backtest_equity.png"))

if __name__ == "__main__":
    main()
