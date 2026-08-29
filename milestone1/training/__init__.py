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
        "MODEL_SIZE_CONFIGS",
        "CheckpointMismatchError",
        "adjust_dropout",
        "build_optimizer_and_schedulers",
        "build_swa_components",
        "find_existing_checkpoint",
        "load_or_create_model",
        "save_training_checkpoint",
        "train",
    ]
except ImportError:
    # Allow importing subpackages (e.g., scaling/) without torch
    __all__ = []
