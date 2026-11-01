# Data Split Notes: Leakage Audit & Purged Manifests

> **TL;DR:** The original benchmark had a massive data leakage issue (90.7% of test events shared price windows with training events). We created 3 clean, zero-leakage purged split manifests with an embargo buffer so the team has honest benchmarks to train and test on.

---

### 1. What was broken in the original split?

In `data/dataloader.py`, the original repo concatenated all 12 event folders (2008–2024) and did a simple 60/20/20 slice by list position:
* Every event is timestamped to **08:30 AM** of its date.
* Input is 35 bars before 8:30 AM; target is the next 140 bars ($\approx 2$ trading days).
* Because different releases happen on the same day (e.g., Jobless Claims + CPI on the same morning), one ended up in Train and the other in Test.

**The Leakage Numbers:**
* **82.1%** of test events shared the exact same calendar date as a training event.
* **98.8%** of test events sat directly inside a training target price window.
* *Result:* The baseline was essentially predicting price curves it had already seen during training.

---

### 2. The 3 Purged Manifest Options (`splits/manifests/`)

We implemented purged block splitting with a $\pm 140$-bar quarantine embargo (boundary events are dropped so no price windows touch). All 3 options have **0.00% date and price leakage**:

| Manifest File | Strategy | Train / Vali / Test Events | Best Used For |
| :--- | :--- | :---: | :--- |
| `purged_split_c1_macro_eras.json` | **Chronological Walk-Forward** (Train: 2008–2017, Vali: 2017–2021, Test: 2021–2024) | 6,167 / 2,500 / 2,921 | **Standard recommendation.** Traditional forward-in-time generalization into modern high-inflation data. |
| `purged_split_c2_iconic_years.json` | **Regime Holdout** (Holds out 4 entire iconic years: 2008 Crash, 2017 Calm, 2020 COVID, 2022 Inflation) | 7,556 / 1,404 / 2,549 | **Great for MoE / Volatility testing.** Directly tests if text helps during panic years while staying quiet during calm years. |
| `purged_split_c3_retrodiction_2008.json` | **Reverse-Chronological Stress Test** (Train: Modern 2011–2024 $\to$ Test: 2008 Financial Crisis) | 10,666 / 737 / 106 | **Stress test.** Prevents the model from cheating on the upward bull-market drift; tests if it can predict a real $-38\%$ market crash. |

---

### 3. How to load them in PyTorch

We wrote a drop-in loader in `data/purged_dataloader.py`:

```python
from data.purged_dataloader import build_purged_dataloaders

train_loader, val_loader, test_loader = build_purged_dataloaders(
    manifest_path="splits/manifests/purged_split_c1_macro_eras.json",
    series_path="data/series/SP500.csv",
    event_dir="data/event",
    batch_size=16  # or 64 on AWS
)
```
