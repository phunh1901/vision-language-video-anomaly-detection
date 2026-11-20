"""V21 native cosine: LR-only pilot after run_yxfrilug, with flow/error auditing."""
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUN_MODE = "full"
# Optional: old run directory, or the attached partial result ZIP. Reuses only
# validated completed full datasets; missing datasets still train/evaluate normally.
RECOVERY_FROM = None
# Example (same Kaggle session): "/kaggle/working/vadclip_v21/run_unazcppm"
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
INCLUDE_PREDICTIONS = True
STOP_ON_FAILURE = True
# Native forward has no Poincare head; no radial norm conditioning is applied.
GEOMETRY_CONFIG = {"encoder_precision": "float32"}
TRAIN_CONFIG = {
    "native_cosine_control": True,
    "initialization": "fresh_downstream",
    "data_protocol": "official_train_final_epoch",
    "epochs": 30,
    "learning_rate": 1e-5,
    "learning_rate_by_dataset": {"ucf": 1e-5, "xd": 2e-5},
    "weight_decay": 5e-5,
    "microbatch_size": 64,
    "gradient_clip": 1.0,
    "amp": False,
    "pilot_epochs": 3,
    "no_decay_scalars_bias_norm": True,
    "optimizer_betas": (0.9, 0.999),
    "learning_health": True,
    "health_batches_per_epoch": 2,
    "health_first_epoch_batches": 20,
    "audit_data_flow": True,
    "verify_encoder_flow": True,
    "amp_initial_scale": 1.0,
    "amp_max_retries": 8,
    # Retained for comparison metadata; actual native loss coefficient/reduction
    # comes directly from the pinned dataset-specific train source, saved in output.
    "phi": {"ucf": 0.1, "xd": 1e-4},
    "loss": {},
    "reproduction_disclosure": {
        "exact_original_training_recipe": False,
        "model_and_objective": "pinned native VadCLIP forward and extracted original objective",
        "matched_to": "native cosine run_yxfrilug (ZIP6): same model/objective, loader, seed, batch, Adam, WD groups, FP32, clipping and full budget",
        "changed": "only initial LR: UCF 5e-4 -> 1e-5; XD 5e-4 -> 2e-5; passive audit added",
        "claim": "LR hypothesis test; pilot has its own 3-epoch cosine schedule; no quality guarantee",
        "selection": "fixed final epoch; no official test checkpoint selection",
    },
}
TRAIN_MANIFESTS = {
    "ucf": "08503ee0733dc34ca9b955ad8d7656d4de70fbe4903f1a716ca6aff56860683f",
    "xd": "156c0638d4668c4ef7896216108981e6d14cc582637ddad744a22db3718cf157",
}
