"""
fetch_prices.py — Download commodity futures + control variable prices
Uses Yahoo Finance via yfinance. Outputs daily OHLCV CSVs.

Usage:
    python -m src.data_collection.fetch_prices
    python -m src.data_collection.fetch_prices --freq 1h   # hourly (limited history)
"""
import argparse
import os
import sys
import time
from datetime import timedelta

import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    COMMODITY_TICKERS, COMMODITY_PROXIES, CONTROL_TICKERS,
    SAMPLE_START, SAMPLE_END, PRICES_DIR,
)


def fetch_single(ticker: str, label: str, start: str, end: str,
                 interval: str = "1d") -> pd.DataFrame | None:
    """Fetch OHLCV for one ticker."""
    print(f"  [{label}] Fetching {ticker} ({interval})...", end=" ")
    try:
        df = yf.download(
            ticker, start=start, end=end,
            interval=interval, auto_adjust=True, progress=False,
        )
        if df.empty:
            print("⚠ empty")
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "date"
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = [c.lower() for c in df.columns]
        print(f"✓ {len(df)} rows  [{df.index[0].date()} → {df.index[-1].date()}]")
        return df
    except Exception as e:
        print(f"✗ {e}")
        return None


def fetch_all(interval: str = "1d"):
    """Fetch all tickers and save to CSV."""
    os.makedirs(PRICES_DIR, exist_ok=True)

    start_str = SAMPLE_START.isoformat()
    # Add buffer for end date
    end_str = (SAMPLE_END + timedelta(days=7)).isoformat()

    all_groups = {
        "commodity": COMMODITY_TICKERS,
        "proxy":     COMMODITY_PROXIES,
        "control":   CONTROL_TICKERS,
    }

    summary = []
    for group_name, tickers in all_groups.items():
        print(f"\n{'='*50}")
        print(f"  Group: {group_name}")
        print(f"{'='*50}")
        for label, ticker in tickers.items():
            df = fetch_single(ticker, label, start_str, end_str, interval)
            if df is not None:
                suffix = f"_{interval}" if interval != "1d" else ""
                fname = f"{label}{suffix}.csv"
                fpath = os.path.join(PRICES_DIR, fname)
                df.to_csv(fpath)
                summary.append({
                    "label": label, "ticker": ticker, "group": group_name,
                    "rows": len(df), "start": str(df.index[0].date()),
                    "end": str(df.index[-1].date()), "file": fname,
                })
            time.sleep(0.5)  # rate limit courtesy

    # Save summary
    if summary:
        sdf = pd.DataFrame(summary)
        sdf.to_csv(os.path.join(PRICES_DIR, "_fetch_summary.csv"), index=False)
        print(f"\n{'='*50}")
        print(f"  Summary: {len(summary)} tickers fetched")
        print(f"{'='*50}")
        print(sdf.to_string(index=False))

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="1d", choices=["1d", "1h", "5m"])
    args = parser.parse_args()
    fetch_all(interval=args.freq)
