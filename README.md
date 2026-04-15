# trump-policy-market-llm

> **LLM-constructed Trump Policy Shock Index vs market price benchmarks: information beyond what prices can decompose.**
>
> **基于LLM构建的川普政策冲击指数：市场价格之外的结构化信息**

<details>
<summary>🇨🇳 中文说明 / Chinese Documentation</summary>

## 项目概述

本项目包含一个三篇论文系列的数据管线、LLM分类框架和实证分析代码。

**核心问题**：LLM从政策新闻中构建的Trump Policy Shock Index (TPSI)，是否包含市场价格信号（AUD/JPY、VIX、USDT溢价）**未充分反映**的信息？

市场价格是优秀的综合信号，但它只是"温度计"——只能告诉你"发烧了"，不能告诉你"是感染还是中暑"。对商品市场来说，关税（需求萎缩→油价↓）和制裁（供给收缩→油价↑）都让AUD/JPY下跌，但对原油方向相反。TPSI通过LLM的语义理解区分政策类型、传导渠道和地理目标，弥补了市场价格无法自行分解的结构化信息缺口。

### 三篇论文

| 论文 | 资产 | 市场价格基准 | 核心发现 |
|------|------|-------------|---------|
| **Paper 2** (先发) | 黄金、白银、WTI原油 | **AUD/JPY** | TPSI在供给侧渠道（制裁、能源政策）提供AUD/JPY遗漏的增量信息 |
| Paper 1 | SPX, NDX, HSI, HSCEI, HSTECH, CSI300 | VIX | TPSI捕获跨市场传导时序（US→HK→CN） |
| Paper 3 | BTC, ETH, XMR | USDT/CNY OTC溢价 | "Trump Crypto Paradox"：pro-crypto言论 vs 政策不确定性 |

三篇**共享同一个LLM分类管线**（只跑一次），各自在TPSI聚合和计量分析阶段独立执行。

### 为什么用AUD/JPY做基准而不是GPR？

| 维度 | GPR Index (Caldara-Iacoviello) | AUD/JPY |
|------|-------------------------------|---------|
| 类型 | 文本指数（与TPSI同源） | **市场价格**（独立信息源） |
| 挑战性 | "LLM赢关键词计数不意外" | **赢市场价格 = 真正有信息增量** |
| 与商品关联 | 间接 | 直接：AUD=中国商品需求，JPY=避险 |

GPR保留在稳健性检验中。

### 管线架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     数据采集层 Data Ingestion                     │
│                                                                 │
│  Yahoo News ────┐                                               │
│  (EN, 20关键词)  │                                               │
│                 ├──► 原始新闻 ──► URL去重 + 语义去重              │
│  ForexFactory ──┤   ~20,000篇                                   │
│  (经济日历)      │                                               │
│                 │                                                │
│  jin10.com ─────┘                                               │
│  (中文, 稳健性)                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLM统一分类管线（三篇共享，跑一次）                    │
│                                                                 │
│  Stage 1 ─ Haiku 4.5 筛选                                       │
│  │  输入: title + snippet + source + date                       │
│  │  输出: {relevant: bool}  通过率~45%                           │
│  │                                                              │
│  Stage 2 ─ Sonnet 4.6 九字段分类                                 │
│  │  category: tariff | sanctions | energy | monetary |           │
│  │           geopolitical | regulatory | crypto_policy           │
│  │  sentiment: -2 to +2                                         │
│  │  magnitude: low | medium | high                              │
│  │  channel: safe_haven | supply_disruption | dollar_channel |   │
│  │          risk_appetite | direct_commodity | trade_exposure |   │
│  │          tech_decoupling | monetary_transmission |            │
│  │          crypto_regulation | capital_flight                   │
│  │  geo_target: us | china | hk | eu | russia | iran | ...      │
│  │  equity/commodity/crypto_relevant: true | false               │
│  │  prior_coverage: none | some | extensive                     │
│  │                                                              │
│  Robustness ─ DeepSeek V3.2 全量重跑                             │
│  仲裁 ─ Opus 4.6 (Sonnet≠DeepSeek的~8%)                         │
│  审计 ─ Opus 4.6 Look-ahead Bias检测                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 TPSI 构建（每篇论文独立聚合）                      │
│                                                                 │
│  Layer 1: 文章级评分 sentiment × magnitude × relevance门控       │
│  Layer 2: 事件级去重 (category × geo_target 分组取中位数)         │
│  Layer 3: 日级聚合 (中位数 + count + dispersion + ratio)         │
│  Layer 4: 子指数分解 (by category / channel / geo)              │
│  Layer 5: 变体 (衰减 / 来源加权 / 正负分离)                      │
│                                                                 │
│       ┌──────────┬──────────┬──────────┐                        │
│       ▼          ▼          ▼          │                        │
│  TPSI_commodity  TPSI_equity  TPSI_crypto                       │
│   (Paper 2)      (Paper 1)    (Paper 3)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     实证分析（各篇独立）                           │
│                                                                 │
│  Paper 2: 商品  vs AUD/JPY                                      │
│  ├── EGARCH(1,1): TPSI in mean + variance equations             │
│  ├── 7 Specifications (AUD/JPY → TPSI → 子指数 → GPR)           │
│  ├── 分渠道回归: supply vs demand vs direct                      │
│  ├── 事件研究 CAR (Liberation Day, IEEPA裁决)                    │
│  └── TVP-VAR + Granger + Δρ相关性偏移                            │
│                                                                 │
│  Paper 1: 股票  vs VIX                                          │
│  ├── Panel Regression (市场固定效应)                              │
│  ├── Cross-market Granger: US → HK → CN 传导时序                 │
│  └── 分地理: TPSI_china → HSI vs TPSI_us → SPX                  │
│                                                                 │
│  Paper 3: 加密  vs USDT/CNY溢价                                  │
│  ├── GJR-GARCH (非对称波动)                                      │
│  ├── CAViaR (尾部风险)                                           │
│  └── DCC-GARCH + "Trump trade" paradox                          │
└─────────────────────────────────────────────────────────────────┘
```

### 预算

| 组成 | 费用 |
|------|------|
| LLM分类（三篇共享，含2.5轮迭代+缓冲） | ~$115 |
| 价格数据（yfinance等） | $0 |
| 新闻采集（Yahoo, ForexFactory, jin10） | $0 |
| **合计** | **~$115** |

</details>

## Overview

This repository contains the data pipeline, LLM classification framework, and empirical analysis code for a **three-paper research series** examining whether LLM-constructed policy shock indices contain information that **market price benchmarks cannot decompose**.

### Core Question

Market prices are excellent aggregate signals — AUD/JPY captures commodity demand and safe-haven sentiment, VIX captures equity fear, USDT premiums capture capital flight pressure. But they are "thermometers": they tell you the temperature, not the diagnosis. Tariffs (demand contraction → oil↓) and sanctions (supply disruption → oil↑) both push AUD/JPY lower, yet affect crude oil in opposite directions. Our LLM-based Trump Policy Shock Index (TPSI) decomposes policy type, transmission channel, and geographic target — structural information that market prices cannot self-decompose.

### Three Papers

| Paper | Assets | Market Benchmark | Key Contribution |
|-------|--------|-----------------|------------------|
| **Paper 2** (first) | Gold, Silver, WTI Crude | **AUD/JPY** | TPSI adds supply-side information invisible to AUD/JPY |
| Paper 1 | SPX, NDX, HSI, HSCEI, HSTECH, CSI300 | **VIX** | TPSI captures US→HK→CN transmission sequence |
| Paper 3 | BTC, ETH, XMR | **USDT/CNY premium** | "Trump Crypto Paradox": pro-crypto rhetoric vs policy uncertainty |

All three share **one LLM classification pipeline** (run once), with paper-specific TPSI aggregation and econometrics.

### Why AUD/JPY Instead of GPR?

| | GPR Index | AUD/JPY |
|---|-----------|---------|
| Type | Text index (same source as TPSI) | **Market price** (independent) |
| Challenge | "LLM beats keyword counting — expected" | **Beating a market signal = genuine information increment** |
| Commodity link | Indirect | Direct: AUD=China commodity demand, JPY=safe haven |

GPR is retained in robustness checks.

## Pipeline Architecture

```
Data Ingestion
  Yahoo News (EN, 20 keywords) + ForexFactory + jin10.com (CN)
  → ~20,000 articles → URL dedup + semantic dedup
       │
       ▼
Unified LLM Pipeline (shared, run once, ~$115)
  Stage 1: Haiku 4.5 screening (relevant? binary)
  Stage 2: Sonnet 4.6 nine-field classification
           category | sentiment | magnitude | channel |
           geo_target | equity/commodity/crypto_relevant |
           prior_coverage | rationale
  Robustness: DeepSeek V3.2 full re-run
  Arbitration: Opus 4.6 (disagreements ~8%)
  Audit: Opus 4.6 look-ahead bias detection
       │
       ▼
TPSI Construction (paper-specific aggregation)
  Layer 1: Article scoring (sentiment × magnitude × relevance gate)
  Layer 2: Event-level dedup (category × geo → median)
  Layer 3: Daily aggregation (median + count + dispersion)
  Layer 4: Sub-indices (by category / channel / geography)
  Layer 5: Variants (decay / source weighting / ratio)
       │
       ├──► Paper 2: TPSI_commodity → EGARCH vs AUD/JPY
       ├──► Paper 1: TPSI_equity → Panel regression vs VIX
       └──► Paper 3: TPSI_crypto → GJR-GARCH vs USDT premium
```

## Key Design Decisions

**Discrete over continuous**: LLMs classify reliably (6-choose-1) but score poorly (0.73 vs 0.71). Magnitude uses {low, medium, high}, not 0.0–1.0. Commodity relevance uses binary, not continuous weights. This follows BIS WP1294 (Kwon et al. 2025) methodology.

**Median over mean**: Daily TPSI aggregates via median (robust to outliers), with event-level deduplication preventing single events from dominating via high article volume.

**Prior coverage replaces surprise factor**: LLMs cannot reliably assess "market expectations" ex-ante without look-ahead bias. Instead, `prior_coverage` (none/some/extensive) measures the news timeline, entering regressions as a novelty interaction term.

## Repository Structure

```
trump-policy-market-llm/
│
├── README.md
├── requirements.txt
├── LICENSE (MIT)
│
├── config/
│   └── settings.py             # Tickers, keywords, events, constants
│
├── data/
│   ├── raw/
│   │   ├── news/               # Raw news corpus (EN + CN)
│   │   └── prices/             # OHLCV price data
│   └── processed/              # LLM-classified, TPSI daily series
│
├── src/
│   ├── data_collection/
│   │   ├── fetch_prices.py
│   │   ├── fetch_news_yahoo.py
│   │   └── fetch_news_forexfactory.py
│   │
│   ├── llm_pipeline/           # Shared across all three papers
│   │   └── llm_classify.py     # Stage 1+2, build_tpsi()
│   │
│   ├── tpsi/                   # TPSI construction modules
│   │   ├── score.py            # Layer 1: article-level scoring
│   │   ├── aggregate.py        # Layer 2-3: event dedup + daily
│   │   ├── subindex.py         # Layer 4: sub-index decomposition
│   │   └── variants.py         # Layer 5: decay, ratio, weighting
│   │
│   └── analysis/
│       ├── event_study.py
│       ├── egarch.py
│       ├── correlation_shift.py
│       └── tvp_var.py
│
├── results/
├── notebooks/
├── run_pipeline.py
└── docs/
    └── proposals/
```

## Sample Period

November 2024 (Trump election) — March 2026

## Tech Stack

- **Language**: Python 3.11+
- **LLM APIs**: Claude (Anthropic), DeepSeek, Gemini (validation)
- **Market Data**: Yahoo Finance (yfinance), tushare/akshare (China)
- **News Sources**: Yahoo News, ForexFactory, jin10.com
- **Key Libraries**: pandas, arch (GARCH), scipy, statsmodels, numpy

## Budget

| Item | Cost |
|------|------|
| LLM classification (shared, 2.5 iterations + buffer) | ~$115 |
| Market data (yfinance, GPR index) | $0 |
| News scraping | $0 |
| **Total** | **~$115** |

## License

MIT
