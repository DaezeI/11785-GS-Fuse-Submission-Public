try:
    from model.gs_fuse import GSFuse, train_three_stage
    __all__ = ["GSFuse", "train_three_stage"]
except ImportError:
    GSFuse = None
    train_three_stage = None
    __all__ = []
