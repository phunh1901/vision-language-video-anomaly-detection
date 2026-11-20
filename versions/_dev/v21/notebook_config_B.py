"""V21 option B: fresh downstream, three-epoch low-curvature warm-up, 30 epochs total."""
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUN_MODE = "smoke"
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
HALF_BATCH = 32
LATENCY_WARMUP = 3
LATENCY_REPEATS = 3
BOOTSTRAP_REPLICATES = 200
# Keep per-video scores for diagnosis; no source features or CLIP weights in the ZIP.
INCLUDE_PREDICTIONS = True
STOP_ON_FAILURE = True
GEOMETRY_CONFIG = {"dimension": 512, "curvature": 0.05, "curvature_min": 1e-5,
                   "curvature_max": 2.0, "scale": 0.7, "temperature": 0.07,
                   "chunk_size": 64, "max_tangent_norm": 3.0}
TRAIN_CONFIG = {
    "initialization": "fresh_downstream", "epochs": 30, "learning_rate": 1e-4,
    "weight_decay": 5e-5, "gradient_clip": 1.0, "amp": False,
    "calibrate_initial_scale": True, "calibrate_classifier_bias": True,
    "from_scratch": {"warmup_epochs": 3, "curvature_lr": 1e-5,
                     "scale_lr": 1e-5, "end_lr_ratio": 0.01},
    "phi": {"ucf": 0.1, "xd": 1e-4},
    "loss": {"psi": 1e-3, "cone_aperture_scale": 0.1,
             "entail_regularizer": 0.01, "separation_margin": 1.0},
}
TRAIN_MANIFESTS = {
    "ucf": "08503ee0733dc34ca9b955ad8d7656d4de70fbe4903f1a716ca6aff56860683f",
    "xd": "156c0638d4668c4ef7896216108981e6d14cc582637ddad744a22db3718cf157",
}
