import json
import math
import sys
from pathlib import Path

import torch


def checkpoint_evidence(path):
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dicts"]["qformer"]
    projection_norm = torch.linalg.vector_norm(state["cond_proj.weight"].double()).item()
    layer_norm_weights = state["layers.0.ln_self.weight"].float()
    return projection_norm, layer_norm_weights


out = Path(sys.argv[1])
stage2_norm, stage2_weights = checkpoint_evidence(out / "stage2_best_model.pth")
stage3_norm, stage3_weights = checkpoint_evidence(out / "best_model.pth")
stage2_all_ones = bool((stage2_weights == 1).all())
stage3_all_ones = bool((stage3_weights == 1).all())
max_layer_norm_change = (stage3_weights - stage2_weights).abs().max().item()

passed = (
    stage2_norm == 0
    and stage2_all_ones
    and math.isfinite(stage3_norm)
    and stage3_norm > 0
    and math.isfinite(max_layer_norm_change)
    and max_layer_norm_change > 0
    and not stage3_all_ones
)

evidence = {
    "result": "PASS" if passed else "FAIL",
    "stage2_cond_proj_weight_norm": stage2_norm if math.isfinite(stage2_norm) else None,
    "stage3_cond_proj_weight_norm": stage3_norm if math.isfinite(stage3_norm) else None,
    "stage2_ln_self_weight_all_ones": stage2_all_ones,
    "stage3_ln_self_weight_all_ones": stage3_all_ones,
    "ln_self_weight_max_abs_change": (
        max_layer_norm_change if math.isfinite(max_layer_norm_change) else None
    ),
}
print(json.dumps(evidence, indent=2, allow_nan=False))
sys.exit(0 if passed else 1)
