"""
event_study.py — Event Study (Cumulative Abnormal Returns) for Paper 2

Identifies major Trump policy shock events from daily TPSI, then computes
CARs for gold, silver, and WTI around those events.

Methods:
  - Constant Mean Return Model (for short samples / smoke tests)
  - Market Model (OLS vs S&P 500, for full sample)

Usage:
    python -m src.analysis.event_study
    python -m src.analysis.event_study --threshold 1.5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    PRICES_DIR, DATA_PROCESSED,
    EVENT_WINDOW_PRE, EVENT_WINDOW_POST, ESTIMATION_WINDOW,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results"
)

COMMODITIES = ["gold", "silver", "wti"]


# ── Data Loading ─────────────────────────────────────────────

def load_prices() -> dict[str, pd.DataFrame]:
    """Load daily close prices for commodities and S&P 500 (market benchmark)."""
    prices = {}
    for name in COMMODITIES + ["sp500"]:
        path = os.path.join(PRICES_DIR, f"{name}.csv")
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df[["close"]].rename(columns={"close": name})
        prices[name] = df
    return prices


def load_tpsi() -> pd.DataFrame:
    """Load daily TPSI."""
    path = os.path.join(DATA_PROCESSED, "tpsi_daily.csv")
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def build_returns(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all prices and compute log returns."""
    merged = pd.DataFrame()
    for name, df in prices.items():
        merged = merged.join(df, how="outer") if not merged.empty else df.copy()

    # Log returns (percentage)
    returns = np.log(merged / merged.shift(1)) * 100
    returns = returns.dropna()
    return returns


# ── Event Identification ─────────────────────────────────────

def identify_events(tpsi: pd.DataFrame, threshold_sigma: float = 1.0,
                    column: str = "tpsi_composite") -> pd.DataFrame:
    """
    Identify shock events: days where |TPSI| exceeds threshold × std.
    For smoke tests with few data points, also include the most extreme day.
    """
    series = tpsi[column].dropna()
    if len(series) < 3:
        # Too few points for stats — just pick the most extreme
        idx = series.abs().idxmax()
        events = tpsi.loc[[idx]].copy()
        events["event_type"] = "max_shock"
        print(f"  ⚠ Only {len(series)} TPSI days — using most extreme as event")
        return events

    mean = series.mean()
    std = series.std()
    threshold = threshold_sigma * std

    mask = (series - mean).abs() > threshold
    events = tpsi[mask].copy()
    events["event_type"] = np.where(
        events[column] < mean, "negative_shock", "positive_shock"
    )

    # Always include the most extreme day
    extreme_idx = series.abs().idxmax()
    if extreme_idx not in events.index:
        row = tpsi.loc[[extreme_idx]].copy()
        row["event_type"] = "max_shock"
        events = pd.concat([events, row])

    events = events.sort_index()
    print(f"  Identified {len(events)} events (threshold: {threshold_sigma}σ = {threshold:.4f})")
    for d, row in events.iterrows():
        print(f"    {d.date()}: TPSI={row[column]:.4f} ({row['event_type']})")

    return events


# ── Abnormal Return Computation ──────────────────────────────

def compute_car_constant_mean(returns: pd.DataFrame, event_date: pd.Timestamp,
                               commodity: str,
                               pre: int = 5, post: int = 5) -> dict:
    """
    Compute CAR using Constant Mean Return Model.
    Normal return = mean return over all available non-event days.
    Suitable for short samples where market model is infeasible.
    """
    if commodity not in returns.columns:
        return None

    series = returns[commodity].dropna()
    trading_dates = series.index.sort_values()

    # Find event position
    if event_date not in trading_dates:
        # Snap to nearest trading day
        diffs = abs(trading_dates - event_date)
        event_date = trading_dates[diffs.argmin()]

    event_pos = trading_dates.get_loc(event_date)

    # Event window bounds (clamped to available data)
    win_start = max(0, event_pos - pre)
    win_end = min(len(trading_dates) - 1, event_pos + post)
    window_dates = trading_dates[win_start:win_end + 1]

    # Estimation: all days outside the window
    est_mask = ~series.index.isin(window_dates)
    if est_mask.sum() < 3:
        # Not enough estimation days — use full sample mean
        mu = series.mean()
    else:
        mu = series[est_mask].mean()

    # Abnormal returns in the event window
    ar = series[window_dates] - mu
    car = ar.cumsum()

    # Relative day labels
    rel_days = list(range(win_start - event_pos, win_end - event_pos + 1))

    return {
        "event_date": event_date,
        "commodity": commodity,
        "mu_hat": mu,
        "window_start": window_dates[0],
        "window_end": window_dates[-1],
        "ar": ar,
        "car": car,
        "rel_days": rel_days,
        "car_total": car.iloc[-1] if len(car) > 0 else 0,
        "car_post": car.iloc[len(car)//2:].iloc[-1] if len(car) > 1 else 0,
    }


def compute_car_market_model(returns: pd.DataFrame, event_date: pd.Timestamp,
                              commodity: str, benchmark: str = "sp500",
                              est_window: int = 120,
                              pre: int = 10, post: int = 10) -> dict:
    """
    Compute CAR using Market Model (OLS: R_i = α + β R_m + ε).
    Requires sufficient estimation window data.
    """
    if commodity not in returns.columns or benchmark not in returns.columns:
        return None

    both = returns[[commodity, benchmark]].dropna()
    trading_dates = both.index.sort_values()

    if event_date not in trading_dates:
        diffs = abs(trading_dates - event_date)
        event_date = trading_dates[diffs.argmin()]

    event_pos = trading_dates.get_loc(event_date)

    # Need at least est_window days before the event window
    est_end = max(0, event_pos - pre - 1)
    est_start = max(0, est_end - est_window)

    if est_end - est_start < 30:
        print(f"    ⚠ Insufficient estimation window for {commodity} "
              f"({est_end - est_start} days < 30). Falling back to constant mean.")
        return compute_car_constant_mean(returns, event_date, commodity, pre, post)

    # Estimation period OLS
    est_dates = trading_dates[est_start:est_end]
    y = both.loc[est_dates, commodity].values
    x = both.loc[est_dates, benchmark].values
    x_with_const = np.column_stack([np.ones(len(x)), x])

    beta, residuals, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
    alpha, beta_m = beta[0], beta[1]
    sigma = np.std(y - x_with_const @ beta)

    # Event window
    win_start = max(0, event_pos - pre)
    win_end = min(len(trading_dates) - 1, event_pos + post)
    window_dates = trading_dates[win_start:win_end + 1]

    # Abnormal returns
    actual = both.loc[window_dates, commodity]
    expected = alpha + beta_m * both.loc[window_dates, benchmark]
    ar = actual - expected
    car = ar.cumsum()

    rel_days = list(range(win_start - event_pos, win_end - event_pos + 1))

    return {
        "event_date": event_date,
        "commodity": commodity,
        "alpha": alpha,
        "beta": beta_m,
        "sigma": sigma,
        "window_start": window_dates[0],
        "window_end": window_dates[-1],
        "ar": ar,
        "car": car,
        "rel_days": rel_days,
        "car_total": car.iloc[-1] if len(car) > 0 else 0,
        "car_post": car.iloc[len(car)//2:].iloc[-1] if len(car) > 1 else 0,
    }


# ── Main Runner ──────────────────────────────────────────────

def run_event_study(threshold_sigma: float = 1.0, method: str = "auto"):
    """Run full event study and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Event Study: Cumulative Abnormal Returns")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    prices = load_prices()
    tpsi = load_tpsi()
    returns = build_returns(prices)
    print(f"  Returns: {len(returns)} trading days "
          f"[{returns.index[0].date()} → {returns.index[-1].date()}]")
    print(f"  TPSI:    {len(tpsi)} days")

    # Identify events
    print("\n[2/4] Identifying shock events...")
    events = identify_events(tpsi, threshold_sigma=threshold_sigma)

    # Choose method based on data availability
    if method == "auto":
        method = "market_model" if len(returns) > 150 else "constant_mean"
    print(f"\n[3/4] Computing CARs (method: {method})...")

    # Compute CARs
    all_results = []
    for event_date, event_row in events.iterrows():
        print(f"\n  Event: {event_date.date()} "
              f"(TPSI={event_row['tpsi_composite']:.4f}, "
              f"type={event_row['event_type']})")

        for commodity in COMMODITIES:
            # Use smaller window for smoke tests
            pre = min(EVENT_WINDOW_PRE, len(returns) // 4)
            post = min(EVENT_WINDOW_POST, len(returns) // 4)
            pre = max(pre, 1)
            post = max(post, 1)

            if method == "market_model":
                result = compute_car_market_model(
                    returns, event_date, commodity,
                    est_window=ESTIMATION_WINDOW, pre=pre, post=post
                )
            else:
                result = compute_car_constant_mean(
                    returns, event_date, commodity, pre=pre, post=post
                )

            if result:
                result["event_type"] = event_row["event_type"]
                result["tpsi_value"] = event_row["tpsi_composite"]
                all_results.append(result)
                print(f"    {commodity:8s}: CAR[{-pre},{+post}] = "
                      f"{result['car_total']:+.4f}%")

    # Save results
    print(f"\n[4/4] Saving results...")
    save_results(all_results, RESULTS_DIR)

    return all_results


def save_results(results: list, output_dir: str):
    """Save CAR summary table and per-event AR series."""
    if not results:
        print("  ⚠ No results to save")
        return

    # Summary table
    summary_rows = []
    for r in results:
        summary_rows.append({
            "event_date": r["event_date"].date(),
            "event_type": r["event_type"],
            "tpsi_value": r["tpsi_value"],
            "commodity": r["commodity"],
            "window": f"[{r['rel_days'][0]},{r['rel_days'][-1]}]",
            "car_total": round(r["car_total"], 4),
            "car_post_event": round(r["car_post"], 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "event_study_car_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Summary → {summary_path}")
    print(summary.to_string(index=False))

    # Detailed AR/CAR time series for each event × commodity
    detail_frames = []
    for r in results:
        df = pd.DataFrame({
            "date": r["ar"].index,
            "rel_day": r["rel_days"][:len(r["ar"])],
            "event_date": r["event_date"].date(),
            "commodity": r["commodity"],
            "ar": r["ar"].values,
            "car": r["car"].values,
        })
        detail_frames.append(df)

    detail = pd.concat(detail_frames, ignore_index=True)
    detail_path = os.path.join(output_dir, "event_study_car_detail.csv")
    detail.to_csv(detail_path, index=False)
    print(f"  Detail  → {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Study (CAR)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Sigma threshold for event identification")
    parser.add_argument("--method", choices=["auto", "constant_mean", "market_model"],
                        default="auto")
    args = parser.parse_args()
    run_event_study(threshold_sigma=args.threshold, method=args.method)
