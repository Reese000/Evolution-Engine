"""Training pipeline modules.

Re-exports all public symbols so callers can do:
    from training import train, load_or_create_model, ...

Note: torch-dependent imports are lazily loaded so that non-torch
subpackages (like scaling/) can be imported without requiring torch.
"""

try:
    from .checkpoint import find_existing_checkpoint, save_training_checkpoint
    from .loop import train
    from .model_factory import MODEL_SIZE_CONFIGS, CheckpointMismatchError, load_or_create_model
    from .schedule import adjust_dropout, build_optimizer_and_schedulers, build_swa_components

    __all__ = [
        "train",
        "load_or_create_model",
        "MODEL_SIZE_CONFIGS",
        "CheckpointMismatchError",
        "save_training_checkpoint",
        "find_existing_checkpoint",
        "build_optimizer_and_schedulers",
        "adjust_dropout",
        "build_swa_components",
    ]
except ImportError:
    # Allow importing subpackages (e.g., scaling/) without torch
    __all__ = []
