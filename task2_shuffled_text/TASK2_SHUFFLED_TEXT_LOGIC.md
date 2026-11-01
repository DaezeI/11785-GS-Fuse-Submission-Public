# Task 2 Notes: Gate Diagnosis & Shuffled-Text Control

> **TL;DR:** We tested whether GS-Fuse actually reads economic news or just treats text as static noise. We evaluated the baseline across all 2,329 test events under 3 conditions: **Real News**, **Randomly Scrambled News**, and **No News (Price-Only)**.
>
> **The Bottom Line:** Randomly shuffling news across dates changes MSE by only **0.15%**, and **No News actually beats Real News on directional accuracy** (51.14% vs 49.94%). The baseline gate is functionally "deaf" to date-specific news.

---

### 1. The Numbers (2,329 Test Events Evaluated)

| Condition | Overall MSE | Overall MAE | $H=35$ DHR | $H=70$ DHR | $H=140$ DHR | $H=140$ Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Real News (Matched)** | **`0.000667`** | **`0.016389`** | 51.52% | 48.99% | **49.94%** | **0.5099** |
| **2. Shuffled News (Scrambled)** | **`0.000668`** | **`0.016402`** | 51.68% | 48.95% | **50.11%** | **0.5161** |
| **3. Zero News (Price Only)** | **`0.000709`** | **`0.017195`** | **52.64%** | **50.75%** | **51.14%** | **0.4225** |

*(Raw metrics are saved in `task2_shuffled_text/outputs/task2_shuffled_text_control.json`)*

---

### 2. Key Takeaways & What's Happening Under the Hood

#### A. The model doesn't care if news is scrambled
* Shuffling news across dates (e.g., pairing a March 2020 emergency COVID rate cut with a calm 2014 price chart) leaves MSE practically untouched ($0.000667 \to 0.000668$).
* If the model were truly learning economic causality (like *surprise rate hikes $\to$ markets fall*), scrambled news would severely hurt performance. Because error stays flat, the model is using text as an unconditioned background bias rather than reading the specific article.

#### B. The gate is nearly constant (~0.52) and injects noise
* Across the test set, the text gate stays locked around $\bar{\alpha}_{\text{text}} \approx 0.524$ on almost every sample, with basically zero correlation ($\approx 0.024$) to whether text actually helped that sample.
* Because the gate blindly blends 52% text even on quiet trading days, it injects random semantic chatter into the price decoder. That's why **Price-Only gets 51.14% directional accuracy while Real News drops to 49.94%**.

#### C. Why did this happen? The "Flat-Line Shortcut"
* In `model/gs_fuse.py`, there is a linear shortcut:
  ```python
  self.ts_residual = nn.Linear(35 * 4, 140 * 4)
  ```
* In the first epoch of training, this shortcut figures out that simply repeating the last known 8:30 AM price forward as a flat line eliminates **~98% of the MSE loss**.
* Because the price shortcut games the loss so easily, backpropagation gradients reaching LLaMA are microscopic. The language model never feels enough gradient pressure to learn real financial vocabulary.

---

### 3. What this means for our project next steps:
1. **Switch to Log Returns ($r_t$):** Predicting return changes instead of raw price levels breaks the flat-line shortcut so the model has to learn real price moves.
2. **Volatility-Conditioned Gate:** Mute the text gate to 0 on quiet days to stop noise injection, and only let the LLM open up when price volatility spikes on real news shocks.
