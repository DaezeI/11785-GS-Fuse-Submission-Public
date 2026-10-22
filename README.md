# GS-Fuse (Customized for CMU 11-785 Intro to Deep Learning)
## Event-Driven S&P 500 Market Forecasting via Multimodal Gated Fusion

> **Course:** CMU 11-785 Intro to Deep Learning (Fall 2026)  
> **Team GitHub:** `https://github.com/DaezeI/11785-GS-Fuse.git`  
> **Shared Google Drive:** [Team Asset Folder (Checkpoints, Data, & Cache)](https://drive.google.com/drive/folders/1ntwrD20YHgKfLea885mOHLBdsXgdGhDC?usp=sharing)  
> **Weights & Biases Project:** `ailodian-carnegie-mellon-university/11785-GS-Fuse-Baseline`  
> **Anchor Paper:** *GS-Fuse: Granger-Supervised Gated Fusion and Multi-Granularity Alignment for Event-Driven Financial Forecasting* (ACM SIGKDD 2026)  

---

## 1. What This Repository Is (Repository Logic Document)

This repository contains our customized, optimized implementation of the **GS-Fuse** multimodal financial forecasting framework, benchmarked against **DLinear** baselines on high-frequency (5-minute) S&P 500 market data.

Given:
1. **Time-Series Input ($L=35$):** 3 hours of pre-event 5-minute OHLC prices ($[B, 4, 35]$).
2. **Text Input:** Raw macroeconomic release statements (FOMC, CPI, Non-Farm Payrolls, etc.).

The system predicts:
* **Future Price Trajectory ($H=35, 70, 140$):** Multi-horizon price curves ($[B, 4, H]$) capturing the market's post-announcement reaction.

---

## 2. Key Engineering Modifications & Architectural Optimizations

The original KDD 2026 repository was designed for multi-GPU Linux clusters and unquantized 8B foundation models. We re-engineered the codebase to run robustly on local GPU hardware (NVIDIA GeForce RTX 4060 Laptop, 8GB VRAM, Windows 11):

### 2.1 Cross-Platform & Windows Dataloader Fixes
* **File:** `data/dataloader.py`
* **Fix:** Resolved POSIX `/` path slicing that threw `IndexError` on Windows backslashes (`\`). Replaced with `os.path.basename(os.path.dirname(...))`.
* **Fast Tree Indexing:** The original author scanned 151,979 files via recursive `os.walk()` per batch (taking 20+ minutes to start). We added an in-memory directory cache (`_event_dir_cache`), reducing dataset initialization from **20+ minutes to 15 seconds**.
* **Multiprocessing Stability:** Set `num_workers=0` across all loaders to prevent Windows `spawn` thread-locking.

### 2.2 4-Bit Foundation Quantization
* **File:** `model/text_encoders.py`, `model/gs_fuse.py`
* **Fix:** Integrated `bitsandbytes` 4-bit NormalFloat (`nf4`) quantization with `torch.float16` compute dtype. Reduced LLaMA-3.2-3B memory footprint from **~13 GB down to ~2.0 GB VRAM**, eliminating out-of-memory crashes.

### 2.3 Persistent NVMe Disk Representation Streaming
* **File:** `model/text_encoders.py`
* **Fix:** The original codebase attempted to retain tens of thousands of uncompressed $[1, 256, 3072]$ token tensors in Python heap memory, accumulating 7+ GB of RAM and crashing the process.
* **Architecture:** Implemented an on-disk MD5 representation cache (`data/cache_llama_emb/{hash}.pt`). Embeddings stream directly from NVMe SSD via `torch.load` in $\sim 0.2\text{ ms}$, keeping system RAM flat under 500 MB and permanently bypassing repeated LLM forward passes.

### 2.4 Multi-Stage Checkpoint Auto-Restoration
* **File:** `model/gs_fuse.py`
* **Fix:** Added automatic stage resumption in `train_three_stage()`. If completed `stage1_best_model.pth` or `stage2_best_model.pth` checkpoints are detected, the pipeline restores weights and skips redundant pre-training epochs.
* **Autograd Safeguard:** Froze quantized `torch.uint8` base weights and enabled gradients exclusively on floating-point adapter modules (gate MLP, linear projections, and Transformer decoder).

### 2.5 Standalone Unimodal Baseline (DLinear)
* **File:** `run_baseline.py`
* **Implementation:** Built a standalone baseline trainer featuring seasonal-trend decomposition (`SeriesDecomp`) and linear horizon projections.
* **Ablation & Refinement:** Integrated `CosineAnnealingLR` decaying from $10^{-3} \to 10^{-5}$, lifting validation directional accuracy from $47.75\% \to 54.71\%$ ($H=35$) and $53.17\% \to 58.02\%$ ($H=140$).
* **MLOps Tracking:** Full Weights & Biases telemetry synced to project `11785-GS-Fuse-Baseline`.

---

## 3. Directory Layout

```text
GS-Fuse/
|-- run_baseline.py               # Standalone DLinear baseline trainer (10 & 50 ep, W&B logging)
|-- train.py                      # Primary 3-stage GS-Fuse training entrypoint
|-- test_pretrained_model.py       # Checkpoint evaluation & financial metric benchmark
|-- TEAM_MIDTERM_MASTER_RESOURCE.md# Rubric-aligned team guide with pre-drafted bullet snippets
|-- CHANGELOG.md                  # Comprehensive modification and audit log
|
|-- model/
|   |-- gs_fuse.py                # Core GSFuse architecture, Granger gate, & training loop
|   |-- ts_encoders.py            # MOMENT-1-large & Kronos time-series encoders
|   |-- text_encoders.py          # 4-bit LLaMA-3.2-3B encoder & NVMe disk caching
|   `-- decoder.py                # 3-layer Transformer forecasting decoder
|
|-- data/
|   |-- dataloader.py             # Event-based dataset loader (cached directory indexing)
|   |-- sliding_window_dataloader.py # Sliding window loader for Stage 1 pre-training
|   |-- series/                   # 5-minute index CSVs (SP500, NASDAQ, DJIA, Treasuries)
|   |-- event/                    # 151,979 macroeconomic release documents
|   `-- cache_llama_emb/          # 28,500+ pre-extracted text representation files (.pt)
|
|-- scripts/
|   `-- cache_moment.py           # Standalone MOMENT offline representation pre-extractor
|
`-- outputs/
    |-- baseline_runs/            # DLinear loss curves, plots, and summary JSONs
    `-- gs_fuse_sp500_scratch/    # Multi-stage GS-Fuse checkpoints and full training logs
```

---

## 4. Quickstart: How to Run the Code

### 4.0 Teammate Setup: Data & Checkpoint Placement (For Running Ablations)
Because raw financial data and multi-gigabyte `.pth` model checkpoints exceed GitHub's 100 MB file limit, download the shared assets from our **[Team Google Drive Asset Folder](https://drive.google.com/drive/folders/1ntwrD20YHgKfLea885mOHLBdsXgdGhDC?usp=sharing)** and extract into the repository root:
* **Raw Datasets:**
  * `data/series/SP500.csv` (16-year 5-min OHLC prices)
  * `data/event/` (151,979 macroeconomic release articles across 6 categories)
* **Pre-Computed LLaMA Cache (Optional but strongly recommended):**
  * `data/cache_llama_emb/` (28,503 `.pt` token tensors — enables fast training and bypasses live LLM GPU passes)
* **Pre-Trained Stage Checkpoints (Required to skip Stages 1 & 2):**
  * `outputs/gs_fuse_sp500_scratch/stage1_best_model.pth` (TS pre-trained weights)
  * `outputs/gs_fuse_sp500_scratch/stage2_best_model.pth` (Text pre-trained weights)

> **Instant Ablation Mode:** When `stage1_best_model.pth` and `stage2_best_model.pth` are present in `outputs/gs_fuse_sp500_scratch/`, `train.py` automatically skips Stages 1 and 2, jumping straight to Stage 3. You can modify the decoder head in `model/gs_fuse.py` and train in under 2 minutes per epoch!

### 4.1 Environment Setup
Ensure Python 3.12 is active with CUDA 12.4 PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes momentfm wandb einops matplotlib pandas tqdm
```

### 4.2 Running the Unimodal Baselines (DLinear)
Run the 50-epoch DLinear baseline across both forecasting horizons:
```bash
# Short-term horizon (H=35 steps ~ 3 hours)
python run_baseline.py --epochs 50 --seq_len 35 --pred_len 35 --use_scheduler --use_wandb

# Long-term horizon (H=140 steps ~ 2 trading days)
python run_baseline.py --epochs 50 --seq_len 35 --pred_len 140 --use_scheduler --use_wandb
```

### 4.3 Running the Full Multimodal Baseline (GS-Fuse)
Launch the 3-stage training pipeline (uses cached text representations automatically):
```bash
python train.py \
  --ts-backbone moment \
  --text-backbone llama \
  --moment-path AutonLab/MOMENT-1-large \
  --text-model-path meta-llama/Llama-3.2-3B \
  --series-id SP500 \
  --seq-len 35 \
  --pred-len 140 \
  --batch-size 8 \
  --stage1-epochs 2 \
  --stage2-epochs 2 \
  --num-epochs 2 \
  --output-path outputs/gs_fuse_sp500_scratch
```

---

## 5. Summary of Baseline Experimental Results

| Model Architecture | Input Modalities | Horizon ($H$) | Training Setup | Val MSE ($\times 10^{-3}$) | Test MSE ($\times 10^{-3}$) | Val DHR | Peak DHR | Key Finding & Gate Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DLinear (Baseline)** | Price Only ($L=35$) | $H=35$ | 10 Epochs, Fixed LR $10^{-3}$ | 3.53 | 3.58 | 47.75% | 51.10% | Underfits; noisy directional betting |
| **DLinear (Refined)** | Price Only ($L=35$) | $H=35$ | 50 Epochs, Cosine Annealing | **3.26** | **3.31** | **54.71%** | **55.91%** | +6.96% DHR gain via smooth LR decay |
| **DLinear (Baseline)** | Price Only ($L=35$) | $H=140$ | 10 Epochs, Fixed LR $10^{-3}$ | 9.48 | 9.55 | 53.17% | 54.40% | Slower convergence on 2-day horizon |
| **DLinear (Refined)** | Price Only ($L=35$) | $H=140$ | 50 Epochs, Cosine Annealing | **9.31** | **9.42** | **58.02%** | **58.87%** | +4.85% DHR gain; captures multi-day drift |
| **GS-Fuse (Stage 1)** | Price Only (Sliding) | $H=140$ | 2 Epochs, MOMENT + Decoder | 5.71 | 5.82 | ~50.2% | 50.8% | Pre-trained continuous temporal feature space |
| **GS-Fuse (Stage 2)** | Event Text Only | $H=140$ | 2 Epochs, LLaMA-3.2 + Decoder | **0.69** | **0.71** | ~53.4% | 54.1% | Text provides sharp volatility cues |
| **GS-Fuse (Stage 3, Best)** | **Multimodal (TS + Text)** | $H=140$ | Full Granger Gate + MGA Alignment | **0.62** | **0.67** | **~56.8%** | **57.5%** | **Text Gate: 51.07% (Paper: 51.55%), +1.13% error reduction** |
| **GS-Fuse (Stage 3, Best)** | **Multimodal (TS + Text)** | $H=70$ | Multi-Horizon Evaluation | **0.40** | **0.40** | **~57.2%** | **58.1%** | **MAE: 0.0131, SMAPE: 4.86%** |
| **GS-Fuse (Stage 3, Best)** | **Multimodal (TS + Text)** | $H=35$ | Multi-Horizon Evaluation | **0.34** | **0.34** | **~58.4%** | **59.2%** | **MAE: 0.0121, SMAPE: 4.36%** |

### Baseline Loss Curves & Directional Accuracy Progression
![DLinear Baseline Loss Curves](figures/baseline_loss_curves_50ep.png)

---

## 6. Proposed Innovation for Final Project: Volatility-Restricted Gate

* **Identified Weakness in Paper:** The published Granger gate opens $\sim 50.7\%$ uniformly across the dataset. During quiet, sideways market regimes, macroeconomic news articles are mostly filler noise, causing the model to overfit and make false breakout predictions.
* **Proposed Novelty:** A **Volatility-Aware Stricter Gate**.
  * **Low Volatility (Calm Days):** Clamp the text gate shut ($g_{\text{text}} \to 0$), forcing the model to rely 100% on historical price momentum.
  * **High Volatility (Shock Days):** Permit the gate to open ($g_{\text{text}} > 0$) only when rolling price volatility ($\sigma_t = \text{std}(\text{returns})$) exceeds an empirical threshold, ensuring text is only consumed when macro announcements actually drive market dislocations.

---

## References

- **[Zhang et al., 2026]** Yang Zhang, En Chun, Ziyun Mao, Yulu Wu, and Jun Wang. *GS-Fuse: Granger-Supervised Gated Fusion and Multi-Granularity Alignment for Event-Driven Financial Forecasting*. Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26), 2026.
- **[Zeng et al., 2023]** Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. *Are Transformers Effective for Time Series?* Proceedings of the AAAI Conference on Artificial Intelligence (AAAI '23), 37(9):11121–11128, 2023.
- **[Goswami et al., 2024]** Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. *MOMENT: A Family of Open Time-Series Foundation Models*. International Conference on Machine Learning (ICML '24), 2024.
- **[Touvron et al., 2023]** Hugo Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv preprint arXiv:2307.09288, 2023.
- **[Wu et al., 2021]** Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*. Advances in Neural Information Processing Systems (NeurIPS '21), 34:22419–22430, 2021.
- **[Zhou et al., 2022]** Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*. International Conference on Machine Learning (ICML '22), pages 27268–27286, 2022.
- **[Jin et al., 2024]** Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, and Qingsong Wen. *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models*. International Conference on Learning Representations (ICLR '24), 2024.
- **[Loshchilov & Hutter, 2017]** Ilya Loshchilov and Frank Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts*. International Conference on Learning Representations (ICLR '17), 2017.
- **[Dettmers et al., 2023]** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems (NeurIPS '23), 2023.
