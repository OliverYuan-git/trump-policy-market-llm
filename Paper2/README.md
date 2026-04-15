# Paper 2: Supply Versus Demand
## LLM-Decomposed Policy Shocks and the AUD/JPY Commodity Signal
## 基于LLM分解的政策冲击与AUD/JPY商品信号

---

<details>
<summary>🇨🇳 中文说明</summary>

### 研究问题

AUD/JPY（澳元/日元）是公认的"商品需求+避险情绪"综合温度计。我们的问题是：LLM从新闻中构建的TPSI，是否包含AUD/JPY**无法反映**的商品定价信息？

### 核心发现（预期）

AUD/JPY有三个盲区：

| 政策事件 | AUD/JPY | WTI原油 | TPSI能区分？ |
|----------|---------|---------|-------------|
| 对中国加关税 | **↓↓** | ↓（需求萎缩） | ✓ tariff + demand |
| 对俄能源制裁 | ↓弱 | **↑**（供给↓） | ✓ sanctions + supply |
| 释放战略石油储备 | 不动 | **↓** | ✓ energy + direct |

关税和制裁都让AUD/JPY下跌，但对原油方向相反。TPSI的增量恰好来自AUD/JPY的盲区——**供给侧渠道和直接政策干预**。

### 方法

- **TPSI构建**：LLM九字段分类 → 离散评分 → 事件去重 → 日频聚合
- **计量**：EGARCH(1,1)，7个specification，核心比较 TPSI+AUD/JPY vs AUD/JPY alone
- **分渠道回归**：供给侧（制裁）vs 需求侧（关税）vs 直接政策（SPR）——同一框架下检验方向相反
- **事件研究**：Liberation Day (2025.4.2)、IEEPA裁决 (2026.2.20)
- **稳健性**：GPR作为替代基准、DeepSeek/Gemini多模型、多种聚合方法

### 三种商品

| 商品 | 核心渠道 | 预期TPSI增量 |
|------|---------|-------------|
| **WTI原油** | 供需双向 + 直接干预 | **最大**（AUD/JPY最大盲区） |
| 黄金 | 避险 + 美元反向 | 中等 |
| 白银 | 贵金属 + 工业需求 | 较小 |

### 数据

- 样本期：2024.11 — 2026.3（~370交易日）
- 新闻：~15,000篇英文 + 5,000篇中文
- 价格：COMEX期货 + 上海金/INE原油（地理异质性）

</details>

## Research Question

AUD/JPY is a well-established market-based "thermometer" combining commodity demand (AUD) and safe-haven sentiment (JPY). We ask: **does our LLM-constructed TPSI contain commodity-relevant information that AUD/JPY cannot capture?**

## Key Insight

AUD/JPY has three structural blind spots:

| Policy Event | AUD/JPY | WTI Crude | Can TPSI Distinguish? |
|---|---|---|---|
| China tariff escalation | **↓↓** (AUD crashes) | ↓ (demand contraction) | ✓ tariff + demand channel |
| Russia oil sanctions | ↓ weak | **↑** (supply disruption) | ✓ sanctions + supply channel |
| SPR release | no reaction | **↓** (direct intervention) | ✓ energy + direct channel |

Tariffs and sanctions both push AUD/JPY lower (both are risk-off), but affect crude oil in **opposite directions**. AUD/JPY cannot decompose this; TPSI can. The incremental value of TPSI comes precisely from AUD/JPY's blind spots — **supply-side channels and direct policy intervention**.

## Three Commodities, Three Stories

| Commodity | Ticker | Primary Channels | Expected TPSI Increment vs AUD/JPY |
|-----------|--------|-----------------|-------------------------------------|
| **WTI Crude** | CL=F | supply + demand + direct | **Largest** (AUD/JPY's biggest blind spot) |
| Gold | GC=F | safe_haven + dollar | Moderate |
| Silver | SI=F | safe_haven(weak) + industrial | Smallest |

## Methodology

### TPSI Construction

LLM classifies each article into 9 discrete fields → article-level scoring (`sentiment × magnitude`, discrete values) → event-level deduplication → daily median aggregation → sub-index decomposition by category, channel, geography.

Design follows BIS WP1294 (Kwon et al. 2025): discrete classification over continuous scoring, median over mean, event-level dedup to prevent volume dominance.

### Econometrics

**EGARCH(1,1)** with TPSI in both mean and variance equations:

```
Mean:  r_t = μ + φ₁·TPSI_t + φ₂·(TPSI_t × novelty_t) + β·ΔAUDJPY_t + γ'X_t + ε_t
Var:   ln(σ²_t) = ω + α|z| + γz + β·ln(σ²) + δ₁·|TPSI_t| + δ₂·dispersion_t
```

**7 Specifications** for 3 commodities (21 regressions):

| Spec | Benchmark | TPSI | Tests |
|------|-----------|------|-------|
| A | — | — | Baseline |
| B | ΔAUDJPY | — | Market signal alone |
| C | — | TPSI | TPSI alone |
| D | — | TPSI + novelty | Surprise interaction |
| **E** | **ΔAUDJPY** | **TPSI** | **Core: TPSI increment vs market** |
| F | ΔAUDJPY | TPSI sub-indices | Channel decomposition |
| G | ΔAUDJPY + GPR | TPSI | Kitchen sink (robustness) |

**Key result**: Spec E vs B — does adding TPSI improve AIC by >2?

### Channel Decomposition (Paper 2's core table)

```
r_wti = μ + β_AUDJPY·ΔAUDJPY + β_tariff·TPSI_tariff 
          + β_supply·TPSI_supply + β_direct·TPSI_direct
          + β_dollar·TPSI_dollar + β_risk·TPSI_risk + γ'X + ε
```

Expected: `β_supply > 0` and `β_tariff < 0` — opposite signs within same framework.

### Natural Experiments

| Event | Date | Direction | Gold | WTI |
|-------|------|-----------|------|-----|
| Liberation Day | 2025-04-02 | Negative | ↑ | ↓ |
| IEEPA ruling | 2026-02-20 | Positive | ↓ | ↑ |

## Data

- **Sample**: Nov 2024 — Mar 2026 (~370 trading days)
- **News**: ~15,000 EN + 5,000 CN articles
- **Prices**: COMEX futures (GC=F, SI=F, CL=F), Brent (BZ=F), Shanghai AU9999 & INE SC
- **Benchmark**: AUD/JPY (AUDJPY=X)
- **Controls**: DXY, VIX, US 10Y yield, S&P 500
- **Robustness benchmark**: GPR Index (Caldara-Iacoviello)

## Validation

7-layer evidence chain: human annotation (κ>0.70), cross-model consistency (Sonnet vs DeepSeek r>0.80), GPR correlation (0.25<r<0.65), Granger causality, incremental AIC, coefficient sign consistency, **cross-commodity correlation shift test** (gold-WTI Δρ at TPSI shock days).

## Robustness

- Alternative LLMs (DeepSeek V3.2, Gemini 2.5 Flash)
- GPR as text-based benchmark (Spec G)
- Aggregation variants (mean, ratio, decay λ=0.5/0.7/0.9)
- Sub-sample (pre/post Liberation Day)
- Brent vs WTI; spot/ETF vs futures
- Shanghai vs COMEX (geographic heterogeneity)
- Chinese-language news inclusion/exclusion

## Part of a Three-Paper Series

This paper shares a unified LLM pipeline with:
- **Paper 1** (*One Shock, Three Markets*): equity indices vs VIX
- **Paper 3** (*The Trump Crypto Paradox*): crypto vs USDT/CNY premium

All three test the same thesis — TPSI contains structural information that market-based benchmarks cannot self-decompose — across different asset classes and market microstructures.