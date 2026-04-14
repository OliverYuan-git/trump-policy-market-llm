"""
fetch_news_forexfactory.py — Scrape ForexFactory economic calendar & news
Extracts high-impact events and Trump-related news items.

Usage:
    python -m src.data_collection.fetch_news_forexfactory
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SAMPLE_START, SAMPLE_END, NEWS_DIR


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def scrape_ff_calendar_week(date_str: str) -> list[dict]:
    """
    Scrape ForexFactory calendar for a given week.
    date_str format: 'jan1.2025' (ForexFactory URL convention)
    """
    url = f"https://www.forexfactory.com/calendar?week={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Calendar fetch failed for {date_str}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    events = []

    rows = soup.select("tr.calendar__row")
    current_date = ""
    for row in rows:
        # Date cell
        date_cell = row.select_one("td.calendar__date")
        if date_cell and date_cell.get_text(strip=True):
            current_date = date_cell.get_text(strip=True)

        # Impact
        impact_cell = row.select_one("td.calendar__impact")
        impact = ""
        if impact_cell:
            icon = impact_cell.select_one("span")
            if icon:
                classes = icon.get("class", [])
                if any("high" in c for c in classes):
                    impact = "high"
                elif any("medium" in c for c in classes):
                    impact = "medium"
                elif any("low" in c for c in classes):
                    impact = "low"

        # Currency
        ccy_cell = row.select_one("td.calendar__currency")
        ccy = ccy_cell.get_text(strip=True) if ccy_cell else ""

        # Event name
        event_cell = row.select_one("td.calendar__event")
        event_name = event_cell.get_text(strip=True) if event_cell else ""

        if not event_name:
            continue

        # Actual / Forecast / Previous
        actual   = (row.select_one("td.calendar__actual") or {})
        forecast = (row.select_one("td.calendar__forecast") or {})
        previous = (row.select_one("td.calendar__previous") or {})

        events.append({
            "date_raw": current_date,
            "week": date_str,
            "currency": ccy,
            "impact": impact,
            "event": event_name,
            "actual": actual.get_text(strip=True) if hasattr(actual, 'get_text') else "",
            "forecast": forecast.get_text(strip=True) if hasattr(forecast, 'get_text') else "",
            "previous": previous.get_text(strip=True) if hasattr(previous, 'get_text') else "",
        })

    return events


def generate_week_strings(start, end):
    """Generate ForexFactory week date strings."""
    weeks = []
    current = start
    while current <= end:
        # FF uses 'mon#.yyyy' format e.g. 'jan6.2025'
        month_abbr = current.strftime("%b").lower()
        week_str = f"{month_abbr}{current.day}.{current.year}"
        weeks.append(week_str)
        current += timedelta(weeks=1)
    return weeks


def fetch_calendar(pause: float = 3.0):
    """Fetch full calendar for sample period."""
    os.makedirs(NEWS_DIR, exist_ok=True)
    weeks = generate_week_strings(SAMPLE_START, SAMPLE_END)

    all_events = []
    for i, w in enumerate(weeks):
        print(f"  [{i+1}/{len(weeks)}] Week: {w}", end=" ")
        events = scrape_ff_calendar_week(w)
        all_events.extend(events)
        print(f"→ {len(events)} events")
        time.sleep(pause)

    if all_events:
        df = pd.DataFrame(all_events)
        # Filter to high-impact USD events (most relevant for commodities)
        df_high = df[
            (df["impact"] == "high") | (df["currency"] == "USD")
        ].copy()

        df.to_csv(os.path.join(NEWS_DIR, "ff_calendar_all.csv"), index=False)
        df_high.to_csv(os.path.join(NEWS_DIR, "ff_calendar_high_usd.csv"), index=False)
        print(f"\nTotal: {len(all_events)} events, {len(df_high)} high-impact/USD")

    return all_events


if __name__ == "__main__":
    fetch_calendar()
