"""Verify and redact the private owner export; derive event-paired sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean

EXPECTED = "ff63401d6f5ae4f26fa308658d6945fe472ab3cd807643a028d5132ad3eb0b85"
PINNED = {"train":"5b958e6c15d5c051f2148ae366bb41fe46f4fbf3e17aa0c2cc7d4b8867d05e3c",
          "validation":"9474f81626bf4b489fa9ab6f307e0fd2ba84d9c2b6d5522d43d98fdc2264feef"}
CONTRASTS = {
    "cross_minus_pooled_ungated": ("cross_matched","pooled_matched"),
    "cross_minus_pooled_gated": ("cross_gate_matched","pooled_gate_matched"),
    "pooled_matched_minus_no_change": ("pooled_matched","no_change"),
    "gated_minus_ungated_cross": ("cross_gate_matched","cross_matched"),
    "gated_minus_ungated_pooled": ("pooled_gate_matched","pooled_matched"),
}
CONSOLE = {
    "no_change":(9.950,170.530,0.0), "frozen_yields":(11.830,226.534,40.5),
    "pooled_matched":(11.787,224.864,40.5), "pooled_previous_statement":(11.785,224.760,40.5),
    "cross_matched":(12.200,237.349,40.5), "cross_previous_statement":(12.196,237.204,40.5),
    "cross_zero_query_inference":(11.857,227.203,40.5), "cross_fixed_train_mean_query_inference":(11.822,225.968,40.5),
    "pooled_gate_matched":(11.810,225.776,40.5), "pooled_fixed_half_matched":(11.807,225.662,40.5),
    "pooled_gate_previous_statement":(11.809,225.731,40.5), "cross_gate_matched":(11.976,230.879,40.5),
    "cross_fixed_half_matched":(12.003,231.659,40.5), "cross_gate_previous_statement":(11.974,230.822,40.5),
}


def percentile(values, fraction):
    at = (len(values)-1)*fraction
    i = int(at)
    return values[i]*(1-(at-i))+values[min(i+1,len(values)-1)]*(at-i)


def intervals(runs):
    # Pair all arms on each statement, average up front over the *same* three seeds.
    by_model = {arm: [r["validation_metrics"][arm]["event_mae_bp"] for r in runs]
                for arm in {arm for pair in CONTRASTS.values() for arm in pair}}
    populations = [set(scores) for rows in by_model.values() for scores in rows]
    if not all(s==populations[0] for s in populations) or len(populations[0])!=12:
        raise ValueError("Incomplete paired statements")
    statements = sorted(populations[0])
    rng = random.Random(42)
    samples = [[rng.randrange(12) for _ in range(12)] for _ in range(20000)]
    out = {}
    for contrast,(left,right) in CONTRASTS.items():
        per_event = [mean(a[g]-b[g] for a,b in zip(by_model[left],by_model[right])) for g in statements]
        estimates = sorted(mean(per_event[i] for i in ids) for ids in samples)
        out[contrast] = {"left":left,"right":right,"statement_delta_mean_bp":mean(per_event),
            "descriptive_95_percentile_range_bp":[percentile(estimates,.025),percentile(estimates,.975)],
            "statements_left_lower":sum(v<0 for v in per_event),"statements_left_higher":sum(v>0 for v in per_event),
            "statements_tied":sum(v==0 for v in per_event)}
    return out


def make_report(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError("Unexpected owner metrics export")
    doc = json.loads(raw)
    if (doc["split_sha256"] != PINNED or doc["synthetic_fixture"] or doc["test_accessed"]
        or doc["rows"] != {"train":180,"validation":60} or doc["statements"] != {"train":36,"validation":12,"gate_oof":24}
        or doc["config"]["model"] != "compact-treasury-crossattention-gate-factorial-v1"
        or doc["config"]["max_epochs"] != 10 or doc["config"]["seeds"] != [23,24,25]):
        raise ValueError("Unexpected run configuration")
    runs, aggregate = doc["runs"],doc["mean_validation_metrics"]
    if [r["seed"] for r in runs] != [23,24,25] or set(aggregate) != set(CONSOLE):
        raise ValueError("Unexpected seeds or outcomes")
    for arm, observed in CONSOLE.items():
        m = aggregate[arm]
        if (round(m["mae_bp"],3),round(m["mse_bp_squared"],3),round(m["endpoint_dhr"]*100,1)) != observed:
            raise ValueError("Metrics disagree with the owner console at displayed precision")
        for key,value in m.items():
            if abs(mean(r["validation_metrics"][arm][key] for r in runs)-value) > 1e-12:
                raise ValueError("Seed average mismatch")
    report = {k:doc[k] for k in ("config","device","torch_version","rows","statements","split_sha256",
                                  "synthetic_fixture","test_accessed","mean_validation_metrics")}
    report["original_uploaded_metrics_sha256"] = EXPECTED
    report["teacher_folds"] = [{k:f[k] for k in ("max_fit_market_date","min_predict_market_date")} |
                               {k+"_count":len(f[k]) for k in ("fit_events","predict_events","purged_events")} for f in doc["teacher_folds"]]
    report["seed_metrics"] = []
    for r in runs:
        if any(r[k]["epochs_run"] != 10 or len(r[k]["curve"])!=11
               for k in ("reference_fit","pooled_fit","cross_fit")):
            raise ValueError("Unexpected fit curve")
        if any(len(r[k]) != 10 for k in ("pooled_gate_oof_fit","cross_gate_oof_fit")):
            raise ValueError("Unexpected gate duration")
        report["seed_metrics"].append({k:r[k] for k in ("seed","verification","attention_audit",
            "trainable_gate_parameters_per_arm","trainable_residual_parameters_per_arm","validation_gate_diagnostics",
            "reference_fit","pooled_fit","cross_fit","pooled_gate_oof_fit","cross_gate_oof_fit")} | {
            "validation_metrics":{key:{name:value for name,value in m.items() if not name.startswith("event_")}
                                  for key,m in r["validation_metrics"].items()},
            "paired_aggregate_mae":{key:{name:value for name,value in m.items() if not name.startswith("by_event")}
                                    for key,m in r["paired_mae"].items()}})
    report["paired_bootstrap_sensitivity"] = {
        "sampling":"20,000 draws of 12 validation statements with replacement; each paired difference is averaged over fixed seeds first; linear 2.5/97.5 percentiles.",
        "rng_seed":42, "units":"basis points; positive means left is worse",
        "limits":"Exploratory repeated validation; no temporal blocking, no independent seeds, no correction across contrasts, not a confirmatory interval.",
        "contrasts":intervals(runs)}
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner-metrics",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path(__file__).parent/"metrics.json")
    args=p.parse_args()
    report=make_report(args.owner_metrics)
    args.out.write_bytes((json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n").encode())
