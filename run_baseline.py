import os
import sys
import time
import json
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work regardless of nested folder structure
repo_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(repo_dir, "data")) and os.path.exists(os.path.join(repo_dir, "GS-FUSE-main", "data")):
    repo_dir = os.path.join(repo_dir, "GS-FUSE-main")

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


def train_and_eval(
    seq_len=35,
    pred_len=35,
    epochs=50,
    batch_size=128,
    lr=1e-3,
    stride=5,
    config_name="Run_A_H35",
    use_wandb=True,
    wandb_project="11785-GS-Fuse-Baseline",
    wandb_mode="auto"
):
    print(f"\n=======================================================")
    print(f" Starting Baseline Experiment: {config_name}")
    print(f" Config: seq_len={seq_len}, pred_len={pred_len}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"=======================================================")
    
    # Initialize Weights & Biases if requested
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            actual_mode = wandb_mode
            if actual_mode == "auto":
                actual_mode = "online" if getattr(wandb, "api", None) and getattr(wandb.api, "api_key", None) else "offline"
            
            wandb_run = wandb.init(
                project=wandb_project,
                name=config_name,
                mode=actual_mode,
                reinit=True,
                config={
                    "model": "DLinear",
                    "dataset": "S&P 500 (5-min OHLC)",
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "stride": stride,
                    "scheduler": "CosineAnnealingLR (1e-3 -> 1e-5)",
                    "loss_function": "MSE + 0.1*MAE (Eq 39)",
                }
            )
            # Define custom metric step so W&B plots everything vs epoch on X-axis
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")
            print(f"[W&B] Initialized run '{config_name}' in mode: {actual_mode}")
        except Exception as e:
            print(f"[W&B Warning] Could not initialize W&B ({e}). Proceeding without W&B logging.")
            wandb_run = None

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
    # Cosine Annealing decay to stabilize weights and prevent late-epoch oscillation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    train_losses = []
    vali_losses = []
    vali_accs = []
    
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
            if num_batches >= 400: # Fast iteration across 400 batches
                break
                
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation loop: Compute both Validation Loss and Directional Accuracy (%)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_directions = 0
        total_directions = 0
        
        with torch.no_grad():
            for _, _, scaled_s, scaled_p in vali_loader:
                scaled_s = scaled_s.to(device)
                scaled_p = scaled_p.to(device)
                preds = model(scaled_s)
                
                # Val Loss: MSE + 0.1*MAE matching training loss
                batch_val_loss = criterion_mse(preds, scaled_p) + 0.1 * criterion_mae(preds, scaled_p)
                val_loss += batch_val_loss.item()
                val_batches += 1
                
                # Directional Accuracy (channel 0: close price)
                pred_moves = preds[:, 0, -1] - scaled_s[:, 0, -1]
                true_moves = scaled_p[:, 0, -1] - scaled_s[:, 0, -1]
                correct_directions += (torch.sign(pred_moves) == torch.sign(true_moves)).sum().item()
                total_directions += scaled_s.size(0)
                
                if val_batches >= 100:
                    break
                    
        avg_val_loss = val_loss / val_batches
        val_dhr = (correct_directions / total_directions) * 100.0
        vali_losses.append(avg_val_loss)
        vali_accs.append(val_dhr)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val Dir Acc: {val_dhr:.2f}% | LR: {current_lr:.6f}")
        
        # Log metrics to W&B with epoch as explicit step
        if wandb_run is not None:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/directional_accuracy": val_dhr,
                "learning_rate": current_lr,
            })

    # Sample batch for forecast visualization
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

    final_val_loss = vali_losses[-1]
    final_val_acc = vali_accs[-1]
    best_val_loss = min(vali_losses)
    best_val_acc = max(vali_accs)
    print(f"\n[{config_name}] Best Val Loss: {best_val_loss:.5f} | Peak Val Dir Acc: {best_val_acc:.2f}% | Final Acc: {final_val_acc:.2f}%")

    if wandb_run is not None:
        import wandb
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["final_val_loss"] = final_val_loss
        wandb.summary["best_directional_accuracy"] = best_val_acc
        wandb.summary["final_directional_accuracy"] = final_val_acc
        wandb.finish()

    return {
        "config_name": config_name,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "train_losses": train_losses,
        "vali_losses": vali_losses,
        "vali_accs": vali_accs,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
        "sample_preds": sample_preds,
        "sample_targets": sample_targets,
        "sample_inputs": sample_inputs
    }

def main():
    parser = argparse.ArgumentParser(description="Run 11-785 Baseline Experiments with W&B Logging")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per run (default: 50)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="11785-GS-Fuse-Baseline", help="W&B project name")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled", "auto"], default="auto", help="W&B mode")
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    out_dir = os.path.join(repo_dir, "outputs", "baseline_runs")
    os.makedirs(out_dir, exist_ok=True)
    
    # RUN 1 (Configuration A): Horizon = 35
    res_a = train_and_eval(
        seq_len=35,
        pred_len=35,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        config_name=f"Run_A_H35_{args.epochs}ep",
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode
    )
    
    # RUN 2 (Configuration B): Horizon = 140
    res_b = train_and_eval(
        seq_len=35,
        pred_len=140,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        config_name=f"Run_B_H140_{args.epochs}ep",
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode
    )
    
    # Generate Plot 1: 2x2 Homework-style grid (Train vs Val Loss + Val Accuracy)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    epochs_range = range(1, args.epochs + 1)
    
    # Row 1, Col 1: Run A Loss
    axes[0, 0].plot(epochs_range, res_a["train_losses"], label="Train Loss (MSE+MAE)", color="blue")
    axes[0, 0].plot(epochs_range, res_a["vali_losses"], label="Val Loss (MSE+MAE)", color="crimson")
    axes[0, 0].set_title(f"Run A (H=35, {args.epochs} Epochs): Train vs Val Loss", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()
    
    # Row 1, Col 2: Run A Directional Accuracy
    axes[0, 1].plot(epochs_range, res_a["vali_accs"], label="Val Directional Acc (%)", color="forestgreen")
    axes[0, 1].set_title(f"Run A (H=35, {args.epochs} Epochs): Val Directional Accuracy (%)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Directional Accuracy (%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()
    
    # Row 2, Col 1: Run B Loss
    axes[1, 0].plot(epochs_range, res_b["train_losses"], label="Train Loss (MSE+MAE)", color="purple")
    axes[1, 0].plot(epochs_range, res_b["vali_losses"], label="Val Loss (MSE+MAE)", color="darkorange")
    axes[1, 0].set_title(f"Run B (H=140, {args.epochs} Epochs): Train vs Val Loss", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()
    
    # Row 2, Col 2: Run B Directional Accuracy
    axes[1, 1].plot(epochs_range, res_b["vali_accs"], label="Val Directional Acc (%)", color="teal")
    axes[1, 1].set_title(f"Run B (H=140, {args.epochs} Epochs): Val Directional Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Directional Accuracy (%)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()
    
    plt.tight_layout()
    loss_plot_path = os.path.join(out_dir, f"baseline_loss_curves_{args.epochs}ep.png")
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"\n[Saved] Homework-style curves saved to: {loss_plot_path}")
    
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
    plt.title(f"Baseline Forecast vs Ground Truth (S&P 500, Horizon=35, {args.epochs} Epochs)")
    plt.xlabel("Time Step (5-min intervals)")
    plt.ylabel("Normalized S&P 500 Price")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    forecast_plot_path = os.path.join(out_dir, f"baseline_forecast_{args.epochs}ep.png")
    plt.savefig(forecast_plot_path, dpi=200)
    plt.close()
    print(f"[Saved] Forecast plot saved to: {forecast_plot_path}")
    
    # Save Summary JSON
    summary = {
        "dataset": "S&P 500 (5-min OHLC, 331k steps)",
        "model": "DLinear Time-Series Baseline (Decomposition + Linear)",
        "epochs": args.epochs,
        f"run_A_H35_{args.epochs}ep": {
            "best_val_loss": res_a["best_val_loss"],
            "final_val_loss": res_a["final_val_loss"],
            "best_val_accuracy": f"{res_a['best_val_acc']:.2f}%",
            "final_val_accuracy": f"{res_a['final_val_acc']:.2f}%",
            "train_losses": res_a["train_losses"],
            "val_losses": res_a["vali_losses"],
            "val_accuracies": res_a["vali_accs"],
        },
        f"run_B_H140_{args.epochs}ep": {
            "best_val_loss": res_b["best_val_loss"],
            "final_val_loss": res_b["final_val_loss"],
            "best_val_accuracy": f"{res_b['best_val_acc']:.2f}%",
            "final_val_accuracy": f"{res_b['final_val_acc']:.2f}%",
            "train_losses": res_b["train_losses"],
            "val_losses": res_b["vali_losses"],
            "val_accuracies": res_b["vali_accs"],
        }
    }
    json_path = os.path.join(out_dir, f"baseline_summary_{args.epochs}ep.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Metrics summary saved to: {json_path}")
    
    print("\n=======================================================")
    print(" 50-EPOCH BASELINE TRAINING COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
