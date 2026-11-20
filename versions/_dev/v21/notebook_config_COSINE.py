"""V21 canonical notebook: pinned repository cosine recipe with ten crops."""
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUN_MODE = "full"  # smoke: two updates/dataset; pilot: 3 train-only epochs.
RECOVERY_FROM = None  # Fresh run; no previous output ZIP required.
DATASETS = ("ucf", "xd")
UCF_DATA_ROOT = None
UCF_CHECKPOINT = None
XD_DATA_ROOT = None
XD_FEATURE_ZIP = None
XD_CHECKPOINT = None
CLIP_WEIGHTS = None
# Original 10-crop train ZIP or extracted folder is required.
# None = automatically discover complete inputs under /kaggle/input.
# Set explicit paths only if there are multiple copies or nonstandard locations.
# Existing vadclip_*_train_v5.zip contains only one crop/video.
UCF_TRAIN_ROOT = None
XD_TRAIN_ROOT = None
SEED = 234
HALF_BATCH = 64  # UCF per-class batch. XD uses its own batch_size below.
LATENCY_WARMUP = 3
LATENCY_REPEATS = 3
BOOTSTRAP_REPLICATES = 200
INCLUDE_PREDICTIONS = True
INCLUDE_EPOCH_CHECKPOINTS = False  # Epoch 3/6 remain in Kaggle output separately.
STOP_ON_FAILURE = True
GEOMETRY_CONFIG = {"encoder_precision": "float32"}
TRAIN_CONFIG = {
    "native_cosine_control": True,
    "native_repository_recipe": True,
    "initialization": "fresh_downstream",
    "data_protocol": "official_train_final_epoch",
    "epochs": 10,
    "learning_rate": 2e-5,
    "learning_rate_by_dataset": {"ucf": 2e-5, "xd": 1e-5},
    "recipe_by_dataset": {
        "ucf": {"batch_size": 128, "microbatch_size": 128,
                "scheduler_milestones": [4, 8], "scheduler_gamma": .1},
        "xd": {"batch_size": 96, "microbatch_size": 96,
               "scheduler_milestones": [3, 6, 10], "scheduler_gamma": .1},
    },
    "optimizer": "AdamW",
    "weight_decay": .01,  # Default of upstream AdamW, including biases/gains.
    "no_decay_scalars_bias_norm": False,
    "microbatch_size": 128,
    "gradient_clip": None,
    "amp": False,
    "pilot_epochs": 3,
    "checkpoint_epochs": [3, 6],
    "optimizer_betas": (.9, .999),
    "learning_health": True,
    "health_batches_per_epoch": 2,
    "health_first_epoch_batches": 20,
    "audit_data_flow": True,
    "verify_encoder_flow": True,
    "amp_initial_scale": 1.,
    "amp_max_retries": 8,
    "phi": {"ucf": .1, "xd": 1e-4},
    "loss": {},
    "reproduction_disclosure": {
        "exact_original_training_recipe": False,
        "matched_to": "defaults in pinned c41067f repository, not paper hyperparameters",
        "changed_from_zip7": "10 crops; native batch/sampler; uniform temporal means; AdamW; MultiStepLR; native LR/seed/epochs; no clipping",
        "remaining_differences": "NumPy shuffle instead of torch DataLoader RNG; train-only gradient probes/reset RNG; final checkpoint rather than author test selection",
        "claim": "recipe reconstruction, not optimizer-only ablation; released checkpoint training history unknown",
        "selection": "fixed final epoch; no official-test checkpoint selection",
        "precision": "FP32 downstream and frozen CLIP encoder; author environment parity unresolved",
    },
}
# Legacy hashes only; native recipe hashes all original crop files on CPU.
TRAIN_MANIFESTS = {
    "ucf": "08503ee0733dc34ca9b955ad8d7656d4de70fbe4903f1a716ca6aff56860683f",
    "xd": "156c0638d4668c4ef7896216108981e6d14cc582637ddad744a22db3718cf157",
}
