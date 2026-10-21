import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work
repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_dir)
os.chdir(repo_dir)

from data.sliding_window_dataloader import sliding_window_set

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

class SeriesDecomp(nn.Module):
    """Moving average decomposition to extract trend and seasonal components."""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, d, seq_len]
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, self.kernel_size // 2)
        x_padded = torch.cat([front, x, end], dim=-1)
        trend = self.avg(x_padded)
        seasonal = x - trend
        return seasonal, trend

class DLinearBaseline(nn.Module):
    """
    DLinear Baseline (AAAI 2023) - Standard SOTA linear baseline for time-series forecasting.
    Decomposes series into trend and seasonal parts, then forecasts each via linear projection.
    """
    def __init__(self, seq_len, pred_len, d_feat=4, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomp = SeriesDecomp(kernel_size)
        
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, d, seq_len]
        seasonal, trend = self.decomp(x)
        seasonal_pred = self.linear_seasonal(seasonal)  # [B, d, pred_len]
        trend_pred = self.linear_trend(trend)           # [B, d, pred_len]
        return seasonal_pred + trend_pred


def train_and_eval(seq_len=35, pred_len=35, epochs=3, batch_size=128, lr=1e-3, stride=5, config_name="Run_A_H35"):
    print(f"\n=======================================================")
    print(f" Starting Baseline Experiment: {config_name} (seq_len={seq_len}, pred_len={pred_len})")
    print(f"=======================================================")
    
    # Load data
    sw_data = sliding_window_set(
        seq_len=seq_len,
        pred_len=pred_len,
        series_id='SP500',
        series_dir='data/series',
        stride=stride,
        batch_size=batch_size,
        scale=True,
        num_workers=0
    )
    train_loader = sw_data.train_loader
    vali_loader = sw_data.vali_loader
    
    model = DLinearBaseline(seq_len=seq_len, pred_len=pred_len, d_feat=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    train_losses = []
    vali_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for raw_s, raw_p, scaled_s, scaled_p in train_loader:
            scaled_s = scaled_s.to(device)
            scaled_p = scaled_p.to(device)
            
            optimizer.zero_grad()
            preds = model(scaled_s)
            
            # Combined MSE + MAE loss matching GS-Fuse paper eq (39)
            loss = criterion_mse(preds, scaled_p) + 0.1 * criterion_mae(preds, scaled_p)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            if num_batches >= 400: # Fast iteration for meeting demo
                break
                
        avg_train = total_loss / num_batches
        train_losses.append(avg_train)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for _, _, scaled_s, scaled_p in vali_loader:
                scaled_s = scaled_s.to(device)
                scaled_p = scaled_p.to(device)
                preds = model(scaled_s)
                val_loss += criterion_mse(preds, scaled_p).item()
                val_batches += 1
                if val_batches >= 100:
                    break
        avg_val = val_loss / val_batches
        vali_losses.append(avg_val)
        
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_train:.5f} | Val MSE: {avg_val:.5f}")

    # Evaluate Directional Hit Rate (DHR) on a sample
    model.eval()
    sample_preds = None
    sample_targets = None
    sample_inputs = None
    with torch.no_grad():
        for _, _, scaled_s, scaled_p in vali_loader:
            scaled_s = scaled_s.to(device)
            scaled_p = scaled_p.to(device)
            sample_preds = model(scaled_s).cpu().numpy()
            sample_targets = scaled_p.cpu().numpy()
            sample_inputs = scaled_s.cpu().numpy()
            break

    # Endpoint move sign accuracy (Directional Hit Rate)
    # Channel 0: Close price
    pred_moves = sample_preds[:, 0, -1] - sample_inputs[:, 0, -1]
    true_moves = sample_targets[:, 0, -1] - sample_inputs[:, 0, -1]
    dhr = float(np.mean(np.sign(pred_moves) == np.sign(true_moves))) * 100.0

    return {
        "config_name": config_name,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "train_losses": train_losses,
        "vali_losses": vali_losses,
        "final_val_mse": vali_losses[-1],
        "directional_hit_rate": dhr,
        "sample_preds": sample_preds,
        "sample_targets": sample_targets,
        "sample_inputs": sample_inputs
    }

def main():
    out_dir = os.path.join(repo_dir, "outputs", "baseline_runs")
    os.makedirs(out_dir, exist_ok=True)
    
    # RUN 1 (Configuration A): Horizon = 35
    res_a = train_and_eval(seq_len=35, pred_len=35, epochs=3, config_name="Run_A (Horizon=35)")
    
    # RUN 2 (Configuration B): Horizon = 140
    res_b = train_and_eval(seq_len=35, pred_len=140, epochs=3, config_name="Run_B (Horizon=140)")
    
    # Generate Plot 1: Loss curves for both configurations
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(res_a["train_losses"], marker='o', label="Train Loss (MSE+MAE)")
    plt.plot(res_a["vali_losses"], marker='s', label="Val MSE")
    plt.title("Run A: Loss Curves (Horizon H=35)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(res_b["train_losses"], marker='o', color='purple', label="Train Loss (MSE+MAE)")
    plt.plot(res_b["vali_losses"], marker='s', color='orange', label="Val MSE")
    plt.title("Run B: Loss Curves (Horizon H=140)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    loss_plot_path = os.path.join(out_dir, "baseline_loss_curves.png")
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"\n[Saved] Loss curves saved to: {loss_plot_path}")
    
    # Generate Plot 2: Forecast Trajectory vs Ground Truth
    plt.figure(figsize=(10, 4))
    sample_idx = 0
    inp = res_a["sample_inputs"][sample_idx, 0]
    tgt = res_a["sample_targets"][sample_idx, 0]
    pred = res_a["sample_preds"][sample_idx, 0]
    
    t_inp = np.arange(len(inp))
    t_out = np.arange(len(inp), len(inp) + len(tgt))
    
    plt.plot(t_inp, inp, color='black', label="Lookback Context (Input)")
    plt.plot(t_out, tgt, color='blue', label="Actual Ground Truth Future")
    plt.plot(t_out, pred, color='red', linestyle="--", label="Baseline DLinear Forecast")
    plt.title("Baseline Forecast vs Ground Truth (S&P 500, Horizon=35)")
    plt.xlabel("Time Step (5-min intervals)")
    plt.ylabel("Normalized S&P 500 Price")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    forecast_plot_path = os.path.join(out_dir, "baseline_forecast_sample.png")
    plt.savefig(forecast_plot_path, dpi=200)
    plt.close()
    print(f"[Saved] Forecast plot saved to: {forecast_plot_path}")
    
    # Save Summary JSON
    summary = {
        "dataset": "S&P 500 (5-min OHLC, 331k steps)",
        "model": "DLinear Time-Series Baseline (Decomposition + Linear)",
        "run_A_H35": {
            "val_mse": res_a["final_val_mse"],
            "directional_hit_rate": f"{res_a['directional_hit_rate']:.2f}%",
            "train_losses": res_a["train_losses"],
            "val_losses": res_a["vali_losses"],
        },
        "run_B_H140": {
            "val_mse": res_b["final_val_mse"],
            "directional_hit_rate": f"{res_b['directional_hit_rate']:.2f}%",
            "train_losses": res_b["train_losses"],
            "val_losses": res_b["vali_losses"],
        }
    }
    json_path = os.path.join(out_dir, "baseline_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Metrics summary saved to: {json_path}")
    
    print("\n=======================================================")
    print(" BASELINE REPRODUCTION COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

