"""
llm_classify.py — LLM-based Trump Policy Shock Index (TPSI) construction
Two-stage pipeline:
  Stage 1: Haiku screening (is this Trump-policy-relevant?)
  Stage 2: Sonnet classification (category + sentiment + commodity relevance)

Usage:
    python -m src.llm_pipeline.llm_classify --input data/raw/news/yahoo_news_raw.json
    python -m src.llm_pipeline.llm_classify --input data/raw/news/yahoo_news_raw.json --dry-run
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import LLM_CONFIG, TPSI_CATEGORIES, DATA_PROCESSED

# ── Prompts ───────────────────────────────────────────────────

STAGE1_SYSTEM = """You are a financial news classifier. Your task is to determine
whether a news article is related to Trump administration policy actions that
could plausibly affect financial markets (commodities, equities, currencies, crypto).

Respond with ONLY a JSON object:
{"relevant": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}

Policy-relevant includes: tariffs, trade policy, sanctions, executive orders,
energy policy, Fed/monetary pressure, geopolitical actions, regulatory changes.
NOT relevant: personal scandals, campaign rhetoric without action, social media
drama without policy content, routine governance."""

STAGE1_USER = """Classify this article:
Title: {title}
Snippet: {snippet}
Source: {source}
Date: {pub_date_raw}"""

STAGE2_SYSTEM = """You are an expert financial analyst classifying Trump policy news
for a commodity market impact study (gold, silver, WTI crude oil).

Given an article, respond with ONLY a JSON object:
{{
  "category": "<one of: {categories}>",
  "sentiment": <-2 to +2>,
  "commodity_relevance": {{
    "gold": <0.0-1.0>,
    "silver": <0.0-1.0>,
    "wti": <0.0-1.0>
  }},
  "shock_magnitude": <0.0-1.0>,
  "transmission_channel": "<one of: safe_haven, supply_disruption, dollar_channel, risk_appetite, direct_policy>",
  "rationale": "one sentence"
}}

Sentiment scale:
  -2 = strong negative shock (major escalation, severe tariff, new sanctions)
  -1 = mild negative (rhetorical escalation, minor policy tightening)
   0 = neutral / ambiguous
  +1 = mild positive (de-escalation signal, partial rollback)
  +2 = strong positive (deal reached, sanctions lifted, major de-escalation)

Shock magnitude: 0 = routine, 0.5 = notable, 1.0 = unprecedented
Commodity relevance: probability this event meaningfully affects each commodity"""

STAGE2_USER = """Classify this Trump policy article for commodity market impact:
Title: {title}
Snippet: {snippet}
Source: {source}
Date: {pub_date_raw}"""


def _extract_json(text: str) -> dict | None:
    """Robustly extract the first top-level JSON object from model output.
    Local models (Qwen etc.) often wrap JSON in prose or markdown fences."""
    if not text:
        return None
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find first balanced {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def call_ollama(system: str, user: str, model: str,
                max_tokens: int = 500) -> dict | None:
    """Call local Ollama /api/chat with JSON-mode enforced."""
    url = f"{LLM_CONFIG['ollama_host']}/api/chat"
    payload = {
        "model": model,
        "think": False,  # Qwen3 is a thinking model — disable, else content is empty
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "format": "json",  # force JSON output
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return _extract_json(content)
    except Exception as e:
        print(f"    Ollama error: {e}")
        return None


def call_anthropic(system: str, user: str, model: str,
                   max_tokens: int = 500) -> dict | None:
    """Call Anthropic API. Returns parsed JSON response or None."""
    try:
        import anthropic
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _extract_json(response.content[0].text)
    except Exception as e:
        print(f"    API error: {e}")
        return None


def call_llm(system: str, user: str, model: str,
             max_tokens: int = 500) -> dict | None:
    """Provider-agnostic dispatcher."""
    provider = LLM_CONFIG.get("provider", "ollama")
    if provider == "ollama":
        return call_ollama(system, user, model, max_tokens)
    elif provider == "anthropic":
        return call_anthropic(system, user, model, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def stage1_screen(article: dict) -> dict | None:
    """Stage 1: screening for Trump policy relevance."""
    user_msg = STAGE1_USER.format(**article)
    return call_llm(STAGE1_SYSTEM, user_msg,
                    model=LLM_CONFIG["stage1_model"], max_tokens=200)


def stage2_classify(article: dict) -> dict | None:
    """Stage 2: deep classification (category + sentiment + relevance)."""
    cats = ", ".join(TPSI_CATEGORIES)
    system = STAGE2_SYSTEM.format(categories=cats)
    user_msg = STAGE2_USER.format(**article)
    return call_llm(system, user_msg,
                    model=LLM_CONFIG["stage2_model"], max_tokens=500)


def run_pipeline(input_path: str, output_dir: str = None, dry_run: bool = False):
    """Run full 2-stage classification pipeline."""
    if output_dir is None:
        output_dir = DATA_PROCESSED
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles from {input_path}")

    if dry_run:
        articles = articles[:3]
        print(f"DRY RUN: processing only {len(articles)} articles")

    # ── Stage 1: Screening ────────────────────────────────────
    print("\n" + "="*50)
    print("  STAGE 1: Haiku Screening")
    print("="*50)

    stage1_results = []
    for i, art in enumerate(articles):
        print(f"  [{i+1}/{len(articles)}] {art['title'][:60]}...", end=" ")
        result = stage1_screen(art)
        if result:
            art["s1_relevant"]   = result.get("relevant", False)
            art["s1_confidence"] = result.get("confidence", 0)
            art["s1_reason"]     = result.get("reason", "")
            stage1_results.append(art)
            status = "✓ RELEVANT" if art["s1_relevant"] else "✗ skip"
            print(f"{status} ({art['s1_confidence']:.2f})")
        else:
            art["s1_relevant"] = None
            stage1_results.append(art)
            print("⚠ failed")
        time.sleep(LLM_CONFIG["rate_limit_pause"])

    relevant = [a for a in stage1_results if a.get("s1_relevant")]
    print(f"\nStage 1: {len(relevant)}/{len(articles)} articles relevant")

    # Save stage 1
    s1_path = os.path.join(output_dir, "stage1_screened.json")
    with open(s1_path, "w", encoding="utf-8") as f:
        json.dump(stage1_results, f, ensure_ascii=False, indent=2)

    # ── Stage 2: Classification ───────────────────────────────
    print("\n" + "="*50)
    print("  STAGE 2: Sonnet Classification")
    print("="*50)

    stage2_results = []
    for i, art in enumerate(relevant):
        print(f"  [{i+1}/{len(relevant)}] {art['title'][:60]}...", end=" ")
        result = stage2_classify(art)
        if result:
            art["category"]             = result.get("category", "other")
            art["sentiment"]            = result.get("sentiment", 0)
            art["commodity_relevance"]  = result.get("commodity_relevance", {})
            art["shock_magnitude"]      = result.get("shock_magnitude", 0)
            art["transmission_channel"] = result.get("transmission_channel", "")
            art["rationale"]            = result.get("rationale", "")
            print(f"✓ {art['category']} | sent={art['sentiment']}")
        else:
            art["category"] = "error"
            print("⚠ failed")
        stage2_results.append(art)
        time.sleep(LLM_CONFIG["rate_limit_pause"])

    # Save stage 2
    s2_path = os.path.join(output_dir, "stage2_classified.json")
    with open(s2_path, "w", encoding="utf-8") as f:
        json.dump(stage2_results, f, ensure_ascii=False, indent=2)

    # ── Build daily TPSI ──────────────────────────────────────
    print("\n" + "="*50)
    print("  Building TPSI")
    print("="*50)
    build_tpsi(stage2_results, output_dir)

    return stage2_results


def _resolve_date(art: dict) -> str:
    """Resolve article date: prefer pub_date, fallback to parsing pub_date_raw."""
    if art.get("pub_date"):
        return art["pub_date"]
    raw = art.get("pub_date_raw", "").strip().rstrip("·").strip()
    scraped = art.get("scraped_at", "")
    if not raw or not scraped:
        return ""
    ref = datetime.fromisoformat(scraped)
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
    return (ref - delta_map[unit]).strftime("%Y-%m-%d")


def build_tpsi(classified: list, output_dir: str):
    """
    Construct daily Trump Policy Shock Index from classified articles.
    TPSI_t = Σ (sentiment_i × shock_magnitude_i × commodity_relevance_i,c)
    for each commodity c ∈ {gold, silver, wti}
    """
    records = []
    for art in classified:
        if art.get("category") == "error":
            continue
        cr = art.get("commodity_relevance", {})
        sent = art.get("sentiment", 0)
        mag = art.get("shock_magnitude", 0)
        records.append({
            "date": _resolve_date(art),
            "title": art["title"],
            "category": art.get("category", ""),
            "sentiment": sent,
            "magnitude": mag,
            "channel": art.get("transmission_channel", ""),
            "rel_gold":   cr.get("gold", 0),
            "rel_silver": cr.get("silver", 0),
            "rel_wti":    cr.get("wti", 0),
            "tpsi_gold":   sent * mag * cr.get("gold", 0),
            "tpsi_silver": sent * mag * cr.get("silver", 0),
            "tpsi_wti":    sent * mag * cr.get("wti", 0),
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("  ⚠ No articles to build TPSI from")
        return

    # Save article-level
    df.to_csv(os.path.join(output_dir, "tpsi_article_level.csv"), index=False)
    print(f"  Article-level TPSI: {len(df)} articles")

    # ── Aggregate to daily TPSI ──────────────────────────────
    df_dated = df[df["date"] != ""].copy()
    if df_dated.empty:
        print("  ⚠ No parseable dates — skipping daily aggregation")
        return

    df_dated["date"] = pd.to_datetime(df_dated["date"])

    # Daily aggregate: sum of shock scores + article count
    daily = df_dated.groupby("date").agg(
        tpsi_gold=("tpsi_gold", "sum"),
        tpsi_silver=("tpsi_silver", "sum"),
        tpsi_wti=("tpsi_wti", "sum"),
        tpsi_composite=("tpsi_gold", lambda x: x.sum()),  # placeholder
        article_count=("title", "count"),
        avg_sentiment=("sentiment", "mean"),
        avg_magnitude=("magnitude", "mean"),
    ).reset_index()

    # Composite = equal-weight average of three commodity TPSIs
    daily["tpsi_composite"] = (
        daily["tpsi_gold"] + daily["tpsi_silver"] + daily["tpsi_wti"]
    ) / 3

    # Category sub-indices (daily sum by category)
    for cat in df_dated["category"].unique():
        mask = df_dated["category"] == cat
        cat_daily = df_dated[mask].groupby("date")["tpsi_gold"].sum()
        col_name = f"tpsi_{cat}"
        daily = daily.merge(
            cat_daily.rename(col_name).reset_index(),
            on="date", how="left"
        )
        daily[col_name] = daily[col_name].fillna(0)

    daily = daily.sort_values("date").reset_index(drop=True)
    daily_path = os.path.join(output_dir, "tpsi_daily.csv")
    daily.to_csv(daily_path, index=False)
    print(f"  Daily TPSI: {len(daily)} trading days → {daily_path}")
    print(daily.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw news JSON")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.dry_run)
