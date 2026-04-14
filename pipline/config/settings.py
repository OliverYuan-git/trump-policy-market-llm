"""
Project Configuration — Paper 2: Commodities
LLM-Constructed Trump Policy Shock Indicators and
Heterogeneous Impacts on Precious Metals and Energy Futures
"""
from datetime import date

# ── Sample Period ──────────────────────────────────────────────
# Short window for local-LLM smoke test (last ~1 month)
SAMPLE_START = date(2026, 3, 11)
SAMPLE_END   = date(2026, 4, 11)

# ── Price Tickers (Yahoo Finance) ─────────────────────────────
# Paper 2 focuses on commodities; P1/P3 tickers kept for reuse
COMMODITY_TICKERS = {
    "gold":   "GC=F",   # COMEX Gold Futures
    "silver": "SI=F",   # COMEX Silver Futures
    "wti":    "CL=F",   # NYMEX WTI Crude Oil Futures
}

# Supplementary: spot / ETF proxies for robustness checks
COMMODITY_PROXIES = {
    "gold_spot":   "XAUUSD=X",
    "silver_spot": "XAGUSD=X",
    "wti_etf":     "USO",
    "gold_etf":    "GLD",
    "silver_etf":  "SLV",
}

# Control variables
CONTROL_TICKERS = {
    "dxy":    "DX-Y.NYB",  # US Dollar Index
    "us10y":  "^TNX",      # 10-Year Treasury Yield
    "vix":    "^VIX",      # CBOE Volatility Index
    "sp500":  "^GSPC",     # S&P 500 (cross-reference)
}

# ── Natural Experiment Event Dates (Paper 2) ──────────────────
EVENTS = {
    "liberation_day":     date(2025, 4, 2),   # "Liberation Day" tariff announcement
    "china_100pct_tariff": date(2025, 10, 1),  # placeholder — update with exact date
    "ieepa_ruling":       date(2026, 2, 1),   # placeholder — update with exact date
}

# Event study windows (trading days)
EVENT_WINDOW_PRE  = 10   # [-10, -1] estimation window tail
EVENT_WINDOW_POST = 10   # [+1, +10]
ESTIMATION_WINDOW = 120  # [-130, -11]

# ── News Sources ──────────────────────────────────────────────
NEWS_SOURCES = {
    "forexfactory": {
        "base_url": "https://www.forexfactory.com/calendar",
        "type": "calendar_and_news",
    },
    "yahoo_news": {
        "search_url": "https://news.search.yahoo.com/search",
        "type": "news_search",
    },
    "jin10": {
        "base_url": "https://www.jin10.com",
        "type": "cn_financial_news",
    },
}

NEWS_KEYWORDS_EN = [
    "Trump tariff", "Trump trade war", "Trump China",
    "Trump sanctions", "Trump executive order",
    "Trump oil", "Trump energy", "Trump gold",
    "Trump IEEPA", "Liberation Day tariff",
    "Trump commodity", "Trump trade policy",
    "Trump Russia sanctions", "Trump Iran sanctions",
    "Trump OPEC", "Trump strategic reserve",
]

NEWS_KEYWORDS_CN = [
    "特朗普关税", "特朗普贸易战", "特朗普制裁",
    "特朗普行政令", "特朗普原油", "特朗普黄金",
    "特朗普大宗商品", "解放日关税",
    "特朗普IEEPA", "特朗普中国",
]

# ── LLM Pipeline ──────────────────────────────────────────────
# provider: "ollama" (local) | "anthropic" (cloud)
LLM_CONFIG = {
    "provider": "ollama",
    # Ollama settings
    "ollama_host": "http://localhost:11434",
    "stage1_model": "qwen3:14b",   # screening (same model; cheap locally)
    "stage2_model": "qwen3:14b",   # classification + sentiment
    # Anthropic fallback (kept for later switch)
    "anthropic_stage1_model": "claude-haiku-4-5-20251001",
    "anthropic_stage2_model": "claude-sonnet-4-6",
    "batch_size": 50,
    "max_retries": 3,
    "rate_limit_pause": 0.0,  # local model — no rate limit needed
}

# TPSI Categories (Paper 2 relevant subset)
TPSI_CATEGORIES = [
    "tariff_trade",        # tariff announcements, trade war escalation/de-escalation
    "sanctions",           # country/entity sanctions (Russia, Iran, China)
    "energy_policy",       # SPR releases, drilling policy, OPEC diplomacy
    "monetary_fiscal",     # Fed pressure, fiscal policy signals
    "geopolitical",        # military actions, diplomatic shifts
    "regulatory",          # executive orders, deregulation
    "other",               # not directly policy-relevant
]

SENTIMENT_SCALE = {
    -2: "strong_negative",   # major escalation / severe shock
    -1: "negative",          # mild escalation
     0: "neutral",           # ambiguous or no clear direction
     1: "positive",          # de-escalation / market-friendly
     2: "strong_positive",   # major de-escalation / resolution
}

# ── File Paths ────────────────────────────────────────────────
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
PRICES_DIR     = os.path.join(DATA_RAW, "prices")
NEWS_DIR       = os.path.join(DATA_RAW, "news")
