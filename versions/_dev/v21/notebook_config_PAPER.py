"""V21 Sanity Fix: disclosed conditioning/numerical changes; not author-exact reproduction."""
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
INCLUDE_PREDICTIONS = True
STOP_ON_FAILURE = True
GEOMETRY_CONFIG = {
    "dimension": 512, "curvature": 0.5, "curvature_min": 0.1, "curvature_max": 1.0,
    "scale": 0.7, "temperature": 0.07, "chunk_size": 64,
    "max_tangent_norm": None, "text_projection": False, "geometry_dtype": "float64",
    "encoder_precision": "float32",
    "curvature_trainable": False, "max_relative_radius": 0.95,
}
TRAIN_CONFIG = {
    "paper_design": False, "sanity_fix": True, "initialization": "fresh_downstream",
    "data_protocol": "official_train_final_epoch", "epochs": 30,
    "learning_rate": 5e-4, "weight_decay": 5e-5,
    "microbatch_size": 64, "gradient_clip": 1.0, "amp": False,
    "pilot_epochs": 3, "single_temperature": True,
    "condition_modality_norms": True, "initialization_probe_videos": 32,
    "no_decay_scalars_bias_norm": True, "optimizer_betas": (0.9, 0.999),
    "learning_health": True, "health_batches_per_epoch": 2,
    "verify_encoder_flow": True,
    "amp_initial_scale": 1.0, "amp_max_retries": 8,
    "calibrate_initial_scale": False, "calibrate_classifier_bias": True,
    "phi": {"ucf": 0.1, "xd": 1e-4},
    "loss": {
        "psi": 1e-3, "cone_aperture_scale": 0.1, "entail_regularizer": 0.01,
        "separation_margin": 1.0,
        # GeometryHead already returns -distance/0.07. CE/evaluator use these units.
        "alignment_loss_temperature": 1.0,
    },
    "reproduction_disclosure": {
        "exact_author_code_verified": False,
        "numerical_precision": "FP32 encoders/fusion and BCE/CE; FP64 manifold/cone/distance regularizers and total loss; AMP disabled after nonfinite temporal gradients in run_wq5sdte1",
        "temperature_interpretation": "single_scale_in_alignment_map_train_and_eval",
        "sanity_changes": [
            "train-only modality norm buffers; median of sampled-video median norms",
            "classifier bias initialized to median TRAIN distance after conditioning",
            "fixed curvature 0.5; no optimizer state for curvature",
            "smooth tangent radial compression with relative-radius cap 0.95",
            "atan2 exterior angle; keep aperture guards and report saturation",
            "zero weight decay on scalars, bias/gain vectors; gradient norm clip 1.0",
        ],
        "text_mapping_interpretation": "d-dimensional ball, no appended zero coordinate",
        "unreported_choices": {
            "curvature_initial": 0.5, "visual_fc_initial": "identity/sqrt(d)",
            "prototype_initial": "normal_std_0.01", "classifier_bias_initial": "train_median_distance",
            "cone_aperture_scale": 0.1, "entail_regularizer": 0.01,
            "separation_margin": 1.0, "entailment_pairs": "video-label MIL top-k",
            "entailment_parent": "text", "temporal_sampling_length": 256,
            "scheduler_eta_min": 0.0, "checkpoint_rule": "fixed_epoch_30_no_test_selection",
        },
        "feature_grid": "locked VadCLIP cached features; not verified as paper stride-8 extraction",
    },
}
TRAIN_MANIFESTS = {
    "ucf": "08503ee0733dc34ca9b955ad8d7656d4de70fbe4903f1a716ca6aff56860683f",
    "xd": "156c0638d4668c4ef7896216108981e6d14cc582637ddad744a22db3718cf157",
}
