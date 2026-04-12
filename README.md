# trump-policy-market-llm

> **LLM-quantified analysis of Trump policy shocks on global equities, commodities, and cryptocurrency markets across US-China-HK geopolitical dimensions.**
>
> **基于大语言模型的川普政策冲击量化分析：美中港地缘维度下的全球股市、大宗商品与加密货币市场影响研究**

<details>
<summary>🇨🇳 中文说明 / Chinese Documentation</summary>

## 项目概述

本项目包含一个三篇论文系列的数据管线、LLM情绪提取框架和实证分析代码。研究聚焦于川普政府的政策通讯（关税、制裁、外交言论、加密货币监管）如何传导至全球金融市场。

我们使用大语言模型（LLM）从新闻文本中构建细粒度、多维度的政策冲击指数（TPSI），并分析其对三类资产的异质性影响：

| 论文 | 资产类别 | 主要标的 |
|------|----------|----------|
| Paper 1 | 股票指数 | S&P 500, Nasdaq 100, 恒生指数, 恒生国企指数, 恒生科技指数, 沪深300 |
| Paper 2 | 大宗商品 | 黄金期货(GC), 白银期货(SI), WTI原油期货(CL) |
| Paper 3 | 加密货币 | Bitcoin(BTC), Ethereum(ETH), Monero(XMR) |

地缘政治分析维度覆盖**美国、中国大陆和香港**，利用各市场独特的制度特征进行跨市场比较。

### 核心管线（Pipeline）

```
┌─────────────────────────────────────────────────────────────────┐
│                     数据采集层 Data Ingestion                     │
│                                                                 │
│  ForexFactory ──┐                                               │
│  (news+calendar)│                                               │
│                 ├──► 原始新闻语料库 ──► 去重 & 时间标准化          │
│  Yahoo News  ───┤   (EN + CN)                                   │
│                 │                                                │
│  金十数据    ───┤                                               │
│  (jin10.com)    │                                               │
│                 │                                                │
│  Truth Social ──┘                                               │
│  (补充,可选)                                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LLM分析管线 LLM Analysis Pipeline                │
│                                                                 │
│  Stage 1 ─ 事件检测 (Haiku 4.5, 低成本初筛)                      │
│  │  输入: 原始新闻  →  输出: 是否川普政策相关 (binary)             │
│  │                                                              │
│  Stage 2 ─ 多维分类 (Sonnet 4.6, 主力模型)                       │
│  │  政策类型: Tariff | Sanction | Diplomatic | Crypto-Reg | Other│
│  │                                                              │
│  Stage 3 ─ 地理标签                                              │
│  │  目标区域: US | China-Mainland | Hong-Kong | EU | Other       │
│  │                                                              │
│  Stage 4 ─ 情绪与强度评分                                        │
│  │  方向: -5 to +5  |  强度: Low / Medium / High                 │
│  │  含Chain-of-Thought推理过程                                   │
│  │                                                              │
│  Stage 5 ─ 指数构建                                              │
│     聚合为日频 Trump Policy Shock Index (TPSI)                   │
│     子指数: TPSI_tariff | TPSI_sanction | TPSI_diplomatic | ...  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    市场数据层 Market Data                         │
│                                                                 │
│  TradingView Premium ──► 价格数据归档 (OHLCV)                    │
│  (定时归档,防数据过期)    ├── 股票指数 (SPX, NDX, HSI, CSI300...) │
│                          ├── 商品期货 (GC, SI, CL)              │
│  Yahoo Finance ──────────┤                                      │
│  (补充+回填)              └── 加密货币 (BTC, ETH, XMR)           │
│                                                                 │
│  Binance API ────────────► 加密货币高频数据 (小时级)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 下游实证分析 Empirical Analysis                    │
│                                                                 │
│  Paper 1: 股票指数                                               │
│  ├── Event Study (CAR)                                          │
│  ├── Panel Regression (市场固定效应)                              │
│  └── Cross-market Granger Causality                             │
│                                                                 │
│  Paper 2: 大宗商品                                               │
│  ├── EGARCH (回报率+波动率建模)                                   │
│  ├── Horse Race vs GPR Index                                    │
│  └── TVP-VAR (跨商品溢出)                                       │
│                                                                 │
│  Paper 3: 加密货币                                               │
│  ├── GJR-GARCH (非对称波动)                                      │
│  ├── CAViaR (尾部风险)                                           │
│  └── DCC-GARCH (跨资产相关性)                                    │
└─────────────────────────────────────────────────────────────────┘
```

### LLM API成本参考（2026年4月）

| 模型 | 输入 ($/1M tokens) | 输出 ($/1M tokens) | 本项目用途 |
|------|-------------------|-------------------|-----------|
| Claude Haiku 4.5 | $1.00 | $5.00 | Stage 1 初筛 |
| Claude Sonnet 4.6 | $3.00 | $15.00 | Stage 2-4 主力分析 |
| Claude Opus 4.6 | $5.00 | $25.00 | 边缘案例验证 |
| DeepSeek V3.2 | $0.14 | $0.28 | Robustness check / 多LLM验证 |

> **预估总成本:** 处理约10,000篇新闻，使用Haiku初筛 + Sonnet精细分析 + Batch API (5折)，总计约 **$50-75 USD**。DeepSeek做同等分析约$3-5。


### 项目结构

```
trump-policy-market-llm/
│
├── README.md                   # 本文件（中英双语）
├── requirements.txt            # Python依赖
├── LICENSE                     # MIT License
│
├── data/
│   ├── raw/
│   │   ├── news/               # 原始新闻语料（EN + CN）
│   │   └── market/             # 原始市场价格数据
│   └── processed/              # LLM标注后的结构化数据
│
├── data_archiver/              # 定时数据归档模块
│
├── src/
│   ├── llm_pipeline/           # 核心LLM分析管线（三篇论文共享）
│   │   ├── prompts/            # Prompt模板
│   │   ├── extraction.py       # LLM API调用（Claude / DeepSeek）
│   │   ├── index_builder.py    # TPSI指数构建
│   │   └── validation.py       # 人工标注一致性验证
│   │
│   ├── paper1_equities/        # 论文1：股票指数
│   ├── paper2_commodities/     # 论文2：大宗商品
│   └── paper3_crypto/          # 论文3：加密货币
│
├── notebooks/                  # 探索性分析
├── results/                    # 输出表格和图表
└── docs/proposals/             # 研究提案
```

</details>

## Overview

This repository contains the data pipeline, LLM sentiment extraction framework, and empirical analysis code for a three-paper research series examining how Trump administration policy communications (tariffs, sanctions, diplomatic rhetoric, crypto regulation) propagate through global financial markets.

We employ large language models to construct granular, multi-dimensional policy shock indices from news text, and analyze their heterogeneous impact across three asset classes:

| Paper | Asset Class | Instruments |
|-------|-------------|-------------|
| Paper 1 | Equity Indices | S&P 500, Nasdaq 100, Hang Seng, HSCEI, HSTECH, CSI 300 |
| Paper 2 | Commodities | Gold futures (GC), Silver futures (SI), WTI Crude Oil (CL) |
| Paper 3 | Cryptocurrency | Bitcoin (BTC), Ethereum (ETH), Monero (XMR) |

The geopolitical analysis spans the **US, mainland China, and Hong Kong**, exploiting each market's unique institutional features for cross-market comparison.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Data Ingestion Layer                       │
│                                                                 │
│  ForexFactory ──┐                                               │
│  (news+calendar)│                                               │
│                 ├──► Raw News Corpus ──► Dedup & Normalization   │
│  Yahoo News  ───┤   (EN + CN)                                   │
│                 │                                                │
│  jin10.com   ───┤                                               │
│  (CN finance)   │                                               │
│                 │                                                │
│  Truth Social ──┘                                               │
│  (supplementary)                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Analysis Pipeline                         │
│                                                                 │
│  Stage 1 ─ Event Detection (Haiku 4.5, low-cost screening)      │
│  │  Input: raw news  →  Output: Trump-policy relevant? (binary) │
│  │                                                              │
│  Stage 2 ─ Multi-Dimensional Classification (Sonnet 4.6)        │
│  │  Policy: Tariff | Sanction | Diplomatic | Crypto-Reg | Other │
│  │                                                              │
│  Stage 3 ─ Geographic Tagging                                   │
│  │  Target: US | China-Mainland | Hong-Kong | EU | Other        │
│  │                                                              │
│  Stage 4 ─ Sentiment & Intensity Scoring                        │
│  │  Direction: -5 to +5  |  Intensity: Low / Medium / High      │
│  │  With Chain-of-Thought reasoning trace                       │
│  │                                                              │
│  Stage 5 ─ Index Construction                                   │
│     Aggregate into daily Trump Policy Shock Index (TPSI)        │
│     Sub-indices: TPSI_tariff | TPSI_sanction | TPSI_diplomatic  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Market Data Layer                           │
│                                                                 │
│  TradingView Premium ──► Price Data Archive (OHLCV)             │
│  (scheduled archival)    ├── Equity indices (SPX, NDX, HSI...)  │
│                          ├── Commodity futures (GC, SI, CL)     │
│  Yahoo Finance ──────────┤                                      │
│  (backfill + supplement)  └── Crypto (BTC, ETH, XMR)           │
│                                                                 │
│  Binance API ────────────► Crypto hourly data (for Paper 3)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Empirical Analysis                           │
│                                                                 │
│  Paper 1: Equity Indices                                        │
│  ├── Event Study (CAR)                                          │
│  ├── Panel Regression (market fixed effects)                    │
│  └── Cross-market Granger Causality                             │
│                                                                 │
│  Paper 2: Commodities                                           │
│  ├── EGARCH (return + volatility modeling)                      │
│  ├── Horse Race vs Caldara-Iacoviello GPR Index                 │
│  └── TVP-VAR (cross-commodity spillover)                        │
│                                                                 │
│  Paper 3: Cryptocurrency                                        │
│  ├── GJR-GARCH (asymmetric volatility)                          │
│  ├── CAViaR (tail risk)                                         │
│  └── DCC-GARCH (cross-asset correlation)                        │
└─────────────────────────────────────────────────────────────────┘
```

## LLM API Cost Reference (April 2026)

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Role in This Project |
|-------|---------------------|----------------------|----------------------|
| Claude Haiku 4.5 | $1.00 | $5.00 | Stage 1 screening |
| Claude Sonnet 4.6 | $3.00 | $15.00 | Stage 2-4 main analysis |
| Claude Opus 4.6 | $5.00 | $25.00 | Edge case validation |
| DeepSeek V3.2 | $0.14 | $0.28 | Robustness check / multi-LLM benchmark |

> **Estimated total cost:** ~10,000 articles processed with Haiku screening + Sonnet analysis + Batch API (50% off) ≈ **$50-75 USD**. DeepSeek equivalent ≈ $3-5.

```
data_archiver/
├── config.yaml              # Instrument list, frequencies, archive paths
├── archive_tradingview.py   # Scheduled TradingView data download
├── archive_crypto.py        # Binance API hourly data archival
├── archive_news.py          # News source scheduled scraping
└── cron_setup.sh            # Crontab configuration (recommended: daily)
```

## Repository Structure

```
trump-policy-market-llm/
│
├── README.md                   # This file (bilingual EN/CN)
├── requirements.txt
├── LICENSE                     # MIT
│
├── data/
│   ├── raw/
│   │   ├── news/               # Raw news corpus (EN + CN)
│   │   └── market/             # Raw market price data
│   └── processed/              # LLM-annotated structured data
│
├── data_archiver/              # Scheduled data archival module
│
├── src/
│   ├── llm_pipeline/           # Core LLM pipeline (shared across papers)
│   │   ├── prompts/            # Prompt templates
│   │   ├── extraction.py       # LLM API calls (Claude / DeepSeek)
│   │   ├── index_builder.py    # TPSI index construction
│   │   └── validation.py       # Human annotation agreement metrics
│   │
│   ├── paper1_equities/        # Paper 1: Equity indices
│   ├── paper2_commodities/     # Paper 2: Commodity futures
│   └── paper3_crypto/          # Paper 3: Cryptocurrency
│
├── notebooks/                  # Exploratory analysis
├── results/                    # Output tables and figures
└── docs/proposals/             # Research proposals
```

## Papers

1. **One Shock, Three Markets** — LLM-based decomposition of Trump policy sentiment and asymmetric equity index responses across the US, Hong Kong, and mainland China.
2. **Beyond the GPR Index** — LLM-constructed Trump policy shock indicators and heterogeneous impacts on gold, silver, and crude oil futures.
3. **The Trump Crypto Paradox** — LLM-quantified policy sentiment, the "Trump trade" hypothesis, and tail risk in digital asset markets.

## Tech Stack

- **Language:** Python 3.11+
- **LLM APIs:** Claude (Anthropic), DeepSeek
- **Market Data:** TradingView Premium, Yahoo Finance, Binance API
- **News Sources:** ForexFactory, Yahoo News, jin10.com, Truth Social (supplementary)
- **Key Libraries:** pandas, statsmodels, arch (GARCH), scipy, matplotlib, requests

## Sample Period

November 2024 (Trump election) — March 2026 (latest available)

## License

MIT
