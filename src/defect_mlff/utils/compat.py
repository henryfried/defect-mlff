"""
Compatibility shims for loading MACE / MACE-Field models across versions.
"""
import inspect


def apply_mace_compat_patch() -> None:
    """
    Compatibility shim for loading MACE / MACE-Field models across versions.

    Fixes three issues that arise depending on PyTorch and MACE version:
    1. ``torch.serialization.add_safe_globals([slice])`` - newer PyTorch
       defaults to ``weights_only=True`` which rejects ``slice`` objects
       embedded in model checkpoints.
    2. ``models.ScaleShiftFieldMACE`` alias - some MACE versions renamed the
       field model class; this registers the alias so older checkpoints load.
    3. ``torch.load`` default ``weights_only=False`` - models serialised with
       older torch/pickle need this to deserialise correctly.

    Call once at the top of any script that loads a MACE model, before
    constructing ``MACECalculator``.
    """
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([slice])
    except Exception:
        pass

    try:
        import mace.modules.models as models
    except Exception:
        return

    if not hasattr(models, "ScaleShiftFieldMACE"):
        target = None
        for name in ("MACEField", "ScaleShiftMACEField", "ScaleShiftMACE", "MACE"):
            obj = getattr(models, name, None)
            if obj is not None and inspect.isclass(obj):
                target = obj
                break
        if target is None:
            for name, obj in vars(models).items():
                if inspect.isclass(obj) and "Field" in name:
                    target = obj
                    break
        if target is not None:
            models.ScaleShiftFieldMACE = target

    import torch
    real_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return real_load(*args, **kwargs)

    torch.load = _torch_load_compat
