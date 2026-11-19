#!/usr/bin/env python3
"""Evaluate existing paired prediction NPZ files specifically on Categories 1-6.

Usage:
    python scripts/evaluate_macro_subgroup.py --npz-path <path_to_predictions.npz>

Filters the N evaluation events to only Categories 1-6 (scheduled macroeconomic
announcements: Jobless Claims, Employment, GDP, FOMC, CPI, PPI) and reports
Real Text vs Shuffled Text vs Article-Free metrics specifically on those macro events.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def evaluate_macro_slice(npz_path):
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    blob = np.load(path, allow_pickle=True)
    
    # Identify keys
    event_types = blob["event_types"].astype(int) if "event_types" in blob else None
    if event_types is None:
        raise KeyError("NPZ missing 'event_types' key")

    # Categories 1 to 6
    macro_mask = np.isin(event_types, [1, 2, 3, 4, 5, 6])
    n_total = len(event_types)
    n_macro = int(np.sum(macro_mask))

    print("=" * 75)
    print("MACRO ANNOUNCEMENT SUBGROUP EVALUATION (Categories 1-6 Only)")
    print(f"Source file : {path.name}")
    print(f"Total Events: {n_total} | Macro Events (1-6): {n_macro} ({n_macro/n_total*100:.1f}%)")
    print(f"Excluded Reuters Events (7-12): {n_total - n_macro} ({(n_total - n_macro)/n_total*100:.1f}%)")
    print("=" * 75)

    if n_macro == 0:
        print("No Categories 1-6 events found in this file.")
        return

    # Check available arrays
    real = blob.get("real_text", None)
    shuffled = blob.get("shuffled_text", None)
    targets = blob.get("targets", None)
    
    # Zero text alternatives
    zero = blob.get("zero_text", blob.get("zeroed_article_tokens", None))
    bypass = blob.get("fusion_bypass_ts_inst", None)

    real_m = real[macro_mask] if real is not None else None
    shuffled_m = shuffled[:, macro_mask] if (shuffled is not None and shuffled.ndim == 4) else (shuffled[macro_mask] if shuffled is not None else None)
    zero_m = zero[macro_mask] if zero is not None else None
    bypass_m = bypass[macro_mask] if bypass is not None else None
    targets_m = targets[macro_mask] if targets is not None else None

    # If targets are available, compute MSE and percentage gains
    if targets_m is not None:
        for h in (35, 70, 140):
            if targets_m.shape[1] < h:
                continue
            
            y_h = targets_m[:, :h, :]
            mse_real = np.mean((real_m[:, :h, :] - y_h) ** 2)

            if shuffled_m.ndim == 4:
                mse_shuff = np.mean([np.mean((shuffled_m[s, :, :h, :] - y_h) ** 2) for s in range(len(shuffled_m))])
            else:
                mse_shuff = np.mean((shuffled_m[:, :h, :] - y_h) ** 2)

            gain_shuff_pct = ((mse_shuff - mse_real) / mse_shuff) * 100.0

            print(f"\nHorizon H={h}:")
            print(f"  Real Text MSE        : {mse_real:.8f}")
            print(f"  Shuffled Text MSE    : {mse_shuff:.8f}")
            print(f"  Gain vs Shuffled (%) : {gain_shuff_pct:+.4f}%")

            if zero_m is not None:
                mse_zero = np.mean((zero_m[:, :h, :] - y_h) ** 2)
                gain_zero_pct = ((mse_zero - mse_real) / mse_zero) * 100.0
                print(f"  Zero-Token MSE       : {mse_zero:.8f} (Gain: {gain_zero_pct:+.4f}%)")

            if bypass_m is not None:
                mse_byp = np.mean((bypass_m[:, :h, :] - y_h) ** 2)
                gain_byp_pct = ((mse_byp - mse_real) / mse_byp) * 100.0
                print(f"  Fusion Bypass MSE    : {mse_byp:.8f} (Gain: {gain_byp_pct:+.4f}%)")
    else:
        # Compute mean absolute difference between real and controls
        print("\n(Targets not embedded in this prediction file; reporting output deviation:)")
        for h in (35, 70, 140):
            if real_m.shape[1] < h:
                continue
            diff_shuff = np.mean(np.abs(real_m[:, :h, :] - (shuffled_m[0, :, :h, :] if shuffled_m.ndim == 4 else shuffled_m[:, :h, :])))
            print(f"Horizon H={h} Mean Absolute Output Shift:")
            print(f"  |Real - Shuffled| : {diff_shuff:.6f}")
            if zero_m is not None:
                diff_zero = np.mean(np.abs(real_m[:, :h, :] - zero_m[:, :h, :]))
                print(f"  |Real - ZeroToken|: {diff_zero:.6f}")
            if bypass_m is not None:
                diff_byp = np.mean(np.abs(real_m[:, :h, :] - bypass_m[:, :h, :]))
                print(f"  |Real - Bypass|   : {diff_byp:.6f}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate macro announcements (Categories 1-6)")
    parser.add_argument("--npz-path", required=True, help="Path to saved evaluation .npz file")
    args = parser.parse_args()
    evaluate_macro_slice(args.npz_path)
