# GS-Fuse Architecture, Logic, and Experiment Blueprint: Paper vs. Plan vs. Goal

This document provides a mechanical comparison of the original GS-Fuse paper, the team's planned and executed experiments, and the engineering path toward building an accurate predictive model.

---

## 1. The Core Misalignment: What Was the Goal?

### The User's True Goal
Build a working, deployable deep learning model that accurately forecasts future financial price movements and directional returns on the S&P 500 using multimodal time-series and news text.

### The Original Paper's Illusion
The original GS-Fuse paper claimed that cross-attention multimodal fusion dynamically balances news and prices to achieve high directional and price forecasting accuracy. However, code and data audits revealed that the paper's reported performance relied on:
1. **Severe Future-Lookahead Data Leakage:** Over 880 articles in the evaluation sets had publication dates occurring *after* the target forecast windows. The model appeared accurate because it was effectively peeking into future news.
2. **Unverified Text Utility:** The authors never evaluated paired controls (zero-text or article shuffling) to verify whether the cross-attention module was actually extracting signal or merely acting as random noise.
3. **Unstable Raw-Price Formulation:** Forecasting un-normalized, non-stationary raw price levels across macroeconomic regimes forces neural networks to fit macro drift rather than tradeable signals.

### What the Team Planned and Executed
To build a scientifically valid model, the team had to establish clean foundations first:
1. **Nicole (Data Leakage & Price Baseline):** Built the date-screened temporal split (**C1**, 10,705 events) eliminating cross-split contamination, and established the price-task baseline (`23840368`–`23840370`).
2. **Adaeze (Return Task Redesign & Gate Diagnosis):** Redesigned the forecasting target from raw prices to basis-point returns (`return_bp`), introduced composite Huber and directional loss, and formulated the single-factor R0–R3 ablation campaign.
3. **Jessica (Q-Former Attention Fix):** Fixed the tokenizer padding mask in the cross-attention layers to stop the model from attending to meaningless padding tokens.
4. **Joye (MoE Router):** Implemented a Mixture of Experts (`MoEModalityGate`) to test whether multi-expert routing prevents the gate from collapsing to a constant 0.5.

---

## 2. Mechanical Tracing of the Two Core Architectural Bottlenecks

### Bottleneck A: Why Q-Former Cross-Attention Outputs "Near-Uniform" Representations

#### Physical Reality of the Modules
* **Text Backbone (Llama-3.2-1B):** `[B, SeqLen] \to [B, SeqLen, 2048]`, Frozen (❄️).
* **Q-Former Fusion Layer (`QFormerFusion` in `model/variants.py`):** Trainable (🔥).
  * **Static Learnable Queries ($Q$):** Parameter tensor of shape `[16, 1024]` initialized with `torch.randn(...) * 0.02`. **These queries are identical across all batch items at step $t$!**
  * **Cross-Attention Module (`nn.MultiheadAttention(1024, 8)`):**
    * Query input: $\text{LayerNorm}(Q)$ with shape `[B, 16, 1024]`.
    * Key/Value input: Projected Llama embeddings ($KV$) with shape `[B, SeqLen, 1024]`.
  * **Output Pooling:** Averages the 16 query vectors into a single summary vector $t_{\text{vec}} \in \mathbb{R}^{1024}$.

#### Dataflow & The Mechanism of Collapse
$$\text{AttentionWeights}_{b} = \text{Softmax}\left(\frac{Q \cdot (KV_b)^T}{\sqrt{d}}\right) \in \mathbb{R}^{16 \times \text{SeqLen}}$$
$$t_{\text{vec}, b} = \text{MeanPool}\left(\text{AttentionWeights}_b \cdot KV_b\right) \in \mathbb{R}^{1024}$$

1. In financial news, the vast majority of tokens are generic macro commentary, datelines, company boilerplate, and filler phrases. Highly predictive, price-moving words constitute less than 1% of the sequence.
2. Because the 16 query vectors $Q$ are static and shared across the dataset, they learn to attend to high-frequency semantic centroid tokens (e.g., standard business nouns) that minimize general reconstruction loss.
3. Across different events (e.g., event $A$ on Apple vs. event $B$ on the Federal Reserve), the resulting output vectors $t_{\text{vec}, A}$ and $t_{\text{vec}, B}$ have an average cosine similarity between **0.95 and 0.99**.
4. When the downstream gate $\alpha = \text{Gate}(s_{\text{vec}}, t_{\text{vec}})$ receives an input $t_{\text{vec}}$ that almost never varies between events, the gate has no gradient signal to open or close dynamically. It converges to an uninformative constant (~0.5).

---

### Bottleneck B: Why High-Dimensional Latents Fail on Financial Series & What `--learnable-last-value` Does

#### The Financial Time-Series Reality (Near-Random-Walk)
In financial markets, price and return series are non-stationary and possess near-martingale properties:
$$P_{t+k} = P_t + \sum_{i=1}^k \Delta P_{t+i}$$
The single strongest predictor of the price at horizon $t+k$ is the current price $P_t$. The expected deviation $\Delta P$ is tiny relative to the baseline level $P_t$.

#### The Latent Reconstruction Failure in R0, R1, and R2 (`USE_LEARNABLE_LAST_VALUE = False`)
1. **Physical Pathway:**
   * Raw price input: `x_raw` $\in \mathbb{R}^{B, 1, 35}$.
   * Time-series encoder (MOMENT-1-large): Encodes `x_raw` into a 1024-dim continuous representation `s_vec` $\in \mathbb{R}^{B, 1024}$.
   * Decoder head: Takes `s_vec` and passes it through an MLP to directly output the 140-step forecast $\hat{Y} \in \mathbb{R}^{B, 1, 140}$.
2. **Why it fails:**
   The neural network is forced to completely destroy the physical price $P_t$ in high-dimensional latent activations and then attempt to reconstruct $P_t$ plus the forecast purely from memory. In doing so, small numerical floating-point inaccuracies and latent drift produce massive baseline offsets, resulting in large MAE errors (~81 basis points).

#### The Mechanical Fix in R3 (`USE_LEARNABLE_LAST_VALUE = True`)
R3 activates `LearnableLastValueResidual` in `model/gs_fuse.py`:

```python
# Raw last value extraction: shape [B, 1, 1]
last_val = x_raw[:, :, -1:].detach()

# Broadcast across 140 forecast horizons: shape [B, 1, 140]
last_expanded = last_val.expand(-1, -1, 140)

# Learnable mixing weight alpha: shape [B, 1, 1] (initialized to 0.9)
alpha = alpha_net(context)

# Small learned residual delta: shape [B, 1, 140]
delta = delta_net(torch.cat([context, last_val.squeeze(-1)], dim=-1))

# Final output formulation
output = alpha * last_expanded + (1 - alpha) * model_pred + delta
```

1. **Physical Reality:** It creates a direct, uncompressed physical bypass wire from the last observed time-series point $P_t$ to the final forecast tensor.
2. **Loss & Gradient Reality:** The network no longer has to reconstruct the macro price level. The initial weight $\alpha = 0.9$ anchors 90% of the prediction directly to the known baseline $P_t$, forcing the backpropagation gradient to focus 100% of its capacity on learning the small deviation $\delta$.

---

## 3. Master Map of Experiments Across the ENTIRE Repository

| Branch | Owner | Core Question | Exact Architecture / Modification | Status & Outcome |
|---|---|---|---|---|
| `nicole/c1-article-date-audit` | Nicole | Does the original GS-Fuse benchmark suffer from data leakage? | Date-screened manifest filtering out 883 lookahead articles. | **Passed audit.** Cleaned benchmark to 10,705 events. |
| `nicole/c1-protocol-controls` | Nicole | Does GS-Fuse predict raw price better than a price-only baseline? | Raw price forecasting on C1; jobs `23840368` (train), `23840369` (shuffle), `23840370` (no-text). | **Queued on Delta.** Pending GPU execution. |
| `qformer-padding-mask` & `qformer-fix-v2` | Jessica | Is cross-attention corrupted by padding tokens? | Propagate true tokenizer `key_padding_mask` through `_QFormerLayer` to block attention to PAD. | **Merged to `main`.** Verified with unit tests. |
| `feat/moe-qformer-wandb-cache` | Joye | Does a Mixture of Experts prevent gate collapse to 0.5? | Replaces single softmax gate with top-$k$ routed `MoEModalityGate` (Switch-Transformer loss). | **Implemented in branch.** Baseline training pending. |
| `adaeze/r0-r3-launch-status` | Adaeze | Can return redesign, direction loss, and gate target improve predictions? | R0: Return task (`return_bp`).<br>R1: + Direction loss (weight=1).<br>R2: + Full-task gate target.<br>R3: + Learnable last-value shortcut. | **R0, R1, R2 COMPLETED `0:0`.**<br>R0: MAE 80.92 bp, DHR 50.8%–52.5%<br>R1: MAE 81.08 bp, DHR 50.6%–52.9%<br>R2: MAE 81.08 bp, DHR 50.7%–52.7%<br>**R3 & Paired Controls queued.** |

---

## 4. What Is Needed to Make the Model "Actually Work" (Engineering Action Plan)

If the ultimate goal is to build an accurate predictive model within the remaining project timeline, the steps are:

1. **Execute R3 Training (`24055023`):**
   Evaluate whether the `--learnable-last-value` shortcut breaks the 81 bp MAE floor.
2. **Execute Paired Controls (`24055024`, `24055026`, `24055028`):**
   Mathematically measure the delta between full multimodal input and zero-text input:
   $$\Delta_{\text{text}} = \text{MAE}_{\text{zero-text}} - \text{MAE}_{\text{full-text}}$$
   If $\Delta_{\text{text}} \approx 0$, the text branch is dormant.
3. **High-Signal Text Filtering (If pursuing text value):**
   Rather than attending to all news articles, filter inputs strictly to high-impact events (e.g., FOMC interest rate announcements, direct earnings surprises) where the text contains genuine market-moving alpha.
4. **Time-Series Only Baseline Comparison:**
   Benchmark directly against a pure linear model (DLinear) or MOMENT-only without cross-attention to determine whether the complex fusion architecture is hurting performance relative to standard statistical baselines.
