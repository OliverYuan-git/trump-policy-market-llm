"""
fetch_news_yahoo.py — Scrape Trump policy news from Yahoo News search
Collects headline, snippet, source, date, URL for LLM classification.

Usage:
    python -m src.data_collection.fetch_news_yahoo
    python -m src.data_collection.fetch_news_yahoo --max-pages 20
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import NEWS_KEYWORDS_EN, NEWS_DIR


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}
SEARCH_URL = "https://news.search.yahoo.com/search"


def parse_relative_date(raw: str, reference: datetime) -> str:
    """Convert '5 days ago', '10 hours ago' etc. to YYYY-MM-DD date string."""
    raw = raw.strip().rstrip("·").strip()
    m = re.match(r"(\d+)\s+(second|minute|hour|day|week|month)s?\s+ago", raw, re.I)
    if not m:
        return ""
    n, unit = int(m.group(1)), m.group(2).lower()
    delta_map = {
        "second": timedelta(seconds=n),
        "minute": timedelta(minutes=n),
        "hour":   timedelta(hours=n),
        "day":    timedelta(days=n),
        "week":   timedelta(weeks=n),
        "month":  timedelta(days=n * 30),
    }
    return (reference - delta_map[unit]).strftime("%Y-%m-%d")


def make_article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def scrape_yahoo_news(query: str, max_pages: int = 10,
                      pause: float = 2.0) -> list[dict]:
    """Scrape Yahoo News search results for a given query."""
    articles = []
    for page in range(1, max_pages + 1):
        params = {"p": query, "b": (page - 1) * 10 + 1}
        try:
            resp = requests.get(SEARCH_URL, params=params,
                                headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"    Page {page} failed: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("div.NewsArticle") or soup.select("li div.compTitle")

        if not items:
            # Try alternative selectors
            items = soup.select("div[class*='NewsArticle']")
            if not items:
                print(f"    Page {page}: no results found, stopping.")
                break

        for item in items:
            try:
                # Extract link
                a_tag = item.select_one("a[href]")
                if not a_tag:
                    continue
                url = a_tag.get("href", "")

                # Extract title
                title = a_tag.get_text(strip=True)

                # Extract snippet
                snippet_el = item.select_one("p") or item.select_one("span.s-desc")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""

                # Extract source & date
                source_el = item.select_one("span.s-source") or item.select_one("span[class*='source']")
                source = source_el.get_text(strip=True) if source_el else ""

                time_el = item.select_one("span.s-time") or item.select_one("span[class*='time']")
                pub_time = time_el.get_text(strip=True) if time_el else ""

                if not title or len(title) < 10:
                    continue

                articles.append({
                    "id": make_article_id(url),
                    "source": f"yahoo_news:{source}",
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "pub_date_raw": pub_time,
                    "query": query,
                    "scraped_at": datetime.now().isoformat(),
                })
                # Resolve relative date to absolute
                articles[-1]["pub_date"] = parse_relative_date(
                    pub_time, datetime.now()
                )
            except Exception:
                continue

        print(f"    Page {page}: {len(items)} items")
        time.sleep(pause)

    return articles


def fetch_all_news(max_pages: int = 10):
    """Run all keyword queries and deduplicate."""
    os.makedirs(NEWS_DIR, exist_ok=True)
    all_articles = []

    for i, kw in enumerate(NEWS_KEYWORDS_EN):
        print(f"\n[{i+1}/{len(NEWS_KEYWORDS_EN)}] Query: '{kw}'")
        arts = scrape_yahoo_news(kw, max_pages=max_pages)
        all_articles.extend(arts)
        print(f"  → {len(arts)} articles")

    # Deduplicate by URL hash
    seen = set()
    unique = []
    for a in all_articles:
        if a["id"] not in seen:
            seen.add(a["id"])
            unique.append(a)

    print(f"\nTotal: {len(all_articles)} raw → {len(unique)} unique")

    # Save
    if unique:
        df = pd.DataFrame(unique)
        out_path = os.path.join(NEWS_DIR, "yahoo_news_raw.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

        # Also save JSON for LLM pipeline
        json_path = os.path.join(NEWS_DIR, "yahoo_news_raw.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unique, f, ensure_ascii=False, indent=2)

    return unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args()
    fetch_all_news(max_pages=args.max_pages)
