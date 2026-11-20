"""Editable standalone Kaggle notebook configuration."""
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUN_MODE = "smoke"  # After successful GPU smoke, set to "full".
DATASETS = ("ucf", "xd")
UCF_DATA_ROOT = None
UCF_CHECKPOINT = None
XD_DATA_ROOT = None
XD_FEATURE_ZIP = None
XD_CHECKPOINT = None
CLIP_WEIGHTS = None
UCF_TRAIN_ROOT = None
XD_TRAIN_ROOT = None
SEED = 42
HALF_BATCH = 32  # 32 normal + 32 abnormal; microbatch=1 with gradient accumulation.
LATENCY_WARMUP = 3
LATENCY_REPEATS = 3
BOOTSTRAP_REPLICATES = 200
INCLUDE_PREDICTIONS = False
STOP_ON_FAILURE = True  # Save partial output; do not spend another hour after a failed dataset.
GEOMETRY_CONFIG = {"dimension": 512, "curvature": 0.5, "scale": 0.7,
                   "temperature": 0.07, "chunk_size": 64, "max_tangent_norm": 3.0}
TRAIN_CONFIG = {"epochs": 16, "learning_rate": 1e-4, "weight_decay": 5e-5,
                "gradient_clip": 1.0, "amp": False, "calibrate_initial_scale": True,
                "two_stage": {"warmup_epochs": 4, "finetune_epochs": 12,
                              "head_lr": 1e-4, "finetune_head_lr": 5e-5,
                              "adapter_lr": 2e-5, "prompt_lr": 2e-5,
                              "curvature_lr": 1e-5, "scale_lr": 1e-5,
                              "finetune_curvature_lr": 5e-6, "finetune_scale_lr": 5e-6,
                              "end_lr_ratio": 0.02},
                "phi": {"ucf": 0.1, "xd": 1e-4},
                "loss": {"psi": 1e-3, "cone_aperture_scale": 0.1,
                         "entail_regularizer": 0.01, "separation_margin": 1.0}}
TRAIN_MANIFESTS = {
    "ucf": "08503ee0733dc34ca9b955ad8d7656d4de70fbe4903f1a716ca6aff56860683f",
    "xd": "156c0638d4668c4ef7896216108981e6d14cc582637ddad744a22db3718cf157",
}
