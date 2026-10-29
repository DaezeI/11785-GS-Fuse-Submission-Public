# -*- coding: utf-8 -*-
"""Optional Weights & Biases tracking. Every call is a no-op unless init() succeeded."""

import os

_run = None

_STEP_KEYS = {
    "stage1": "stage1/step",
    "stage2": "stage2/step",
    "stage3": "stage3/step",
}


def add_cli_args(parser):
    g = parser.add_argument_group("Weights & Biases")
    g.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    g.add_argument("--wandb-project", type=str, default="11785-GS-Fuse")
    g.add_argument("--wandb-entity", type=str, default=None, help="W&B user or team (default: your login)")
    g.add_argument("--wandb-name", type=str, default=None, help="Run name (default: derived from the variant)")
    g.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    g.add_argument("--wandb-tags", type=str, nargs="*", default=None)


def init_from_args(args, default_name):
    if not args.wandb:
        return None
    return init(project=args.wandb_project, entity=args.wandb_entity,
                name=args.wandb_name or default_name, config=vars(args),
                mode=args.wandb_mode, tags=args.wandb_tags, out_dir=args.output_path)


def init(project, entity=None, name=None, config=None, mode="online", tags=None, out_dir=None):
    global _run
    try:
        import wandb
    except ImportError:
        print("[W&B] wandb is not installed; run `pip install wandb`. Continuing without W&B.")
        return None
    _run = wandb.init(project=project, entity=entity, name=name, config=config or {},
                      mode=mode, tags=tags, dir=out_dir, reinit=True)
    for prefix, step_key in _STEP_KEYS.items():
        wandb.define_metric(step_key)
        wandb.define_metric(f"{prefix}/*", step_metric=step_key)
    print(f"[W&B] Logging to {_run.url or _run.dir}")
    return _run


def active():
    return _run is not None


def log(metrics, stage=None, step=None):
    """Log a flat dict. With stage/step, keys are prefixed by the stage and plotted against its step."""
    if _run is None:
        return
    if stage is not None:
        metrics = {f"{stage}/{k}": v for k, v in metrics.items()}
        if step is not None:
            metrics[_STEP_KEYS[stage]] = step
    _run.log(metrics)


def log_images(paths, prefix="interpret"):
    if _run is None:
        return
    import wandb
    images = {f"{prefix}/{os.path.splitext(os.path.basename(p))[0]}": wandb.Image(p)
              for p in paths if os.path.exists(p)}
    if images:
        _run.log(images)


def summary(values):
    if _run is None:
        return
    for k, v in values.items():
        _run.summary[k] = v


def finish():
    global _run
    if _run is not None:
        _run.finish()
        _run = None
