# ruff: noqa: F821
"""Notebook driver; configuration and embedded sources are defined in earlier cells."""

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from zipfile import ZIP_DEFLATED, ZipFile


def run_dataset(name, runner, config, scripts):
    header = ["from pathlib import Path"]
    for key, value in config.items():
        code = f"Path({str(value)!r})" if isinstance(value, Path) else repr(value)
        header.append(f"{key} = {code}")
    script = scripts / f"{name}.py"
    script.write_text("\n".join(header) + "\n" + runner, encoding="utf-8")
    print(f"\n--- {name} ---", flush=True)
    with (scripts.parent / f"{name}_console.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            [sys.executable, "-u", str(script)], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        ) as process:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"{name} failed; see {name}_console.log")


V0_REFERENCE = {
    "ucf": ("classification", "roc_auc", 0.8801992605166816),
    "xd": ("branch_2", "frame_ap", 0.8450448809004643),
}


def compare(prefix, paths, analysis):
    import numpy as np

    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    reference = reports["v0"]
    expected = 290 if prefix == "ucf" else 800
    for report in reports.values():
        if report["run_kind"] != "full_checkpoint_inference" or report["video_count"] != expected:
            raise ValueError("Full inference report missing official videos")
        for key in ("split_sha256", "ground_truth_sha256", "evaluated_checkpoint_sha256",
                    "visual_block_length", "frames_per_feature", "original_model_sha256"):
            if report[key] != reference[key]:
                raise ValueError(f"Baseline/variant provenance mismatch: {key}")
    branch, metric, target = V0_REFERENCE[prefix]
    observed = reference["metrics"][branch][metric]
    if abs(observed - target) > 0.00005:
        raise ValueError(f"V0 parity failed: {observed} vs {target}; do not interpret V16")
    arrays = {}
    for variant, path in paths.items():
        with np.load(path.with_name("predictions.npz"), allow_pickle=False) as z:
            arrays[variant] = {k: z[k].copy() for k in z.files}
    old, identity, new = (arrays[k] for k in ("v0", "identity", "v16"))
    for changed in (identity, new):
        for key in ("snippet_offsets", "frame_gt", "score_columns", "video_label",
                    "video_id" if prefix == "ucf" else "official_path"):
            if not np.array_equal(old[key], changed[key]):
                raise ValueError(f"Prediction mapping mismatch: {key}")
    if not np.array_equal(old["scores"], identity["scores"]):
        raise ValueError("beta=0 control does not reproduce BOTH V0 branches exactly")
    if not np.array_equal(old["scores"][:, 0], new["scores"][:, 0]):
        raise ValueError("C-Branch changed, invalid A-only experiment")
    gt, offsets = old["frame_gt"].reshape(-1), old["snippet_offsets"]
    if offsets[-1] * 16 != len(gt):
        raise ValueError("Snippet/frame mapping mismatch")
    a0, a1 = old["scores"][:, 1], new["scores"][:, 1]
    ci = analysis.paired_bootstrap(a0, a1, gt, offsets, BOOTSTRAP_REPLICATES, SEED)
    rows = []
    # Report BOTH branches. UCF's primary C score is intentionally unchanged.
    for column, name in enumerate(("C", "A")):
        before = analysis.score_arrays(old["scores"][:, column], gt)
        after = analysis.score_arrays(new["scores"][:, column], gt)
        for measure in ("roc_auc", "ap"):
            bounds = ci["auc_ci95_pp" if measure == "roc_auc" else "ap_ci95_pp"]
            if name == "C":
                bounds = [0.0, 0.0]
            rows.append({"dataset": prefix, "branch": name, "metric": measure,
                         "v0_pct": 100 * before[measure], "v16_pct": 100 * after[measure],
                         "delta_pp": 100 * (after[measure] - before[measure]),
                         "ci95_low_pp": None if bounds is None else bounds[0],
                         "ci95_high_pp": None if bounds is None else bounds[1]})
    if prefix == "ucf":
        for name, native_name in (("C", "classification"), ("A", "alignment")):
            before = reference["metrics"][native_name]["ano_auc"]
            after = reports["v16"]["metrics"][native_name]["ano_auc"]
            rows.append({"dataset": prefix, "branch": name, "metric": "ano_auc",
                         "v0_pct": 100 * before, "v16_pct": 100 * after,
                         "delta_pp": 100 * (after - before),
                         "ci95_low_pp": None, "ci95_high_pp": None})
    false_alarms = []
    for variant, scores in (("v0_A", a0), ("v16_A", a1)):
        false_alarms.extend({"dataset": prefix, "variant": variant, **r}
                           for r in analysis.false_alarm_diagnostics(scores, gt, offsets))
    records = json.loads(paths["v16"].with_name("anchor_diagnostics.json").read_text())
    ids = [r["video_id"] for r in records]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError("Anchor diagnostics must contain exactly one record per video")
    return rows, false_alarms, {
        "identity_exact": True, "C_exact": True, "bootstrap": ci,
        "active_anchor_videos": sum(r["beta"] > 0 for r in records),
        "fallback_videos": sum(r["beta"] == 0 for r in records),
        "mean_beta": float(np.mean([r["beta"] for r in records])),
        "changed_A_snippets": int(np.count_nonzero(a0 != a1)),
    }


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_both():
    if RUN_MODE not in {"smoke", "full"}:
        raise ValueError("RUN_MODE must be smoke or full")
    if not DATASETS or len(set(DATASETS)) != len(DATASETS) or set(DATASETS) - {"ucf", "xd"}:
        raise ValueError("DATASETS must contain ucf and/or xd without duplicates")
    if type(BOOTSTRAP_REPLICATES) is not int or BOOTSTRAP_REPLICATES < 0:
        raise ValueError("Invalid bootstrap count")
    if not INPUT.is_dir():
        raise FileNotFoundError("Attach four original V0 test/checkpoint datasets")
    root = WORK / "vadclip_v16"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run_", dir=root))
    scripts = output / "runtime_scripts"
    scripts.mkdir()
    for filename, source in (("anchor.py", ANCHOR_SOURCE), ("analysis.py", ANALYSIS_SOURCE)):
        (scripts / filename).write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("v16_analysis", scripts / "analysis.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    spec = importlib.util.spec_from_file_location("v16_anchor", scripts / "anchor.py")
    anchor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anchor)
    anchor.validate_config(ANCHOR_CONFIG)
    experiment = {
        "version": "v16_instance_adaptive_normal_anchor", "mode": RUN_MODE,
        "anchor_config": ANCHOR_CONFIG, "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "status": "fixed_exploratory_inference_on_previously_seen_official_test",
        "training": False, "online_causal": False,
        "selection": "no test labels used in anchor or parameter selection",
        "runtime_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in (
            ("anchor", ANCHOR_SOURCE), ("analysis", ANALYSIS_SOURCE),
            ("ucf", UCF_RUNNER), ("xd", XD_RUNNER))},
    }
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print("Output:", output, flush=True)
    shared = {"INPUT": INPUT, "WORK": WORK, "CLIP_WEIGHTS": CLIP_WEIGHTS,
              "SRC_REPO": Path(tempfile.gettempdir()) / "vadclip_xd_c41067f" / "upstream"}
    jobs = {
        "ucf": (UCF_RUNNER, {"DATA_ROOT": UCF_DATA_ROOT, "CHECKPOINT": UCF_CHECKPOINT}),
        "xd": (XD_RUNNER, {"DATA_ROOT": XD_DATA_ROOT, "CHECKPOINT": XD_CHECKPOINT,
                            "FEATURE_ZIP": XD_FEATURE_ZIP, "REFERENCE_MANIFEST": None,
                            "REQUIRE_PACK_MANIFEST": True, "RUN_MODE": RUN_MODE}),
    }
    runs, metrics, alarms = {}, [], []
    started = time.perf_counter()
    for prefix in DATASETS:
        runner, config = jobs[prefix]
        entry = {"status": "running", "reports": {}}
        runs[prefix] = entry
        try:
            paths = {}
            for variant in ("v0", "identity", "v16"):
                destination = output / prefix / variant
                settings = {**ANCHOR_CONFIG, "beta_max": 0.0} if variant == "identity" else ANCHOR_CONFIG
                run_dataset(f"{prefix}_{variant}", runner, {
                    **shared, **config, "VARIANT": variant, "OUTPUT": destination,
                    "SMOKE_ONLY": RUN_MODE == "smoke", "ANCHOR_CONFIG": settings,
                }, scripts)
                base = destination / RUN_MODE
                candidates = [base / "run.json"] if prefix == "ucf" else list(base.glob("run_*/run.json"))
                if len(candidates) != 1 or not candidates[0].is_file():
                    raise ValueError(f"Missing report: {prefix}/{variant}")
                paths[variant] = candidates[0]
                entry["reports"][variant] = str(candidates[0].relative_to(output))
            if RUN_MODE == "full":
                rows, fa, entry["diagnostics"] = compare(prefix, paths, analysis)
                metrics.extend(rows)
                alarms.extend(fa)
            else:
                report = json.loads(paths["v16"].read_text())
                # Smoke estimate excludes downloads/input hashing and is only an estimate.
                if prefix == "ucf":
                    seconds, sampled = report["smoke_seconds"], report["video_count"]
                else:
                    seconds, sampled = report["elapsed_seconds"], report["forward_calls"]
                entry["estimated_three_full_passes_seconds"] = (
                    seconds / sampled * (290 if prefix == "ucf" else 800) * 3)
                entry["estimate_note"] = "Rough smoke extrapolation; video lengths vary; excludes I/O/setup"
                print(prefix, "rough full inference seconds:", entry["estimated_three_full_passes_seconds"])
            entry["status"] = "ok"
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(prefix, entry["error"], flush=True)
        finally:
            (output / "summary.json").write_text(json.dumps({
                **experiment, "runs": runs, "metrics": metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }, indent=2), encoding="utf-8")
    write_csv(output / "comparison.csv", metrics)
    write_csv(output / "false_alarm_diagnostics.csv", alarms)
    for row in metrics:
        print(f"{row['dataset']} {row['branch']} {row['metric']}: "
              f"{row['v0_pct']:.5f}% -> {row['v16_pct']:.5f}% ({row['delta_pp']:+.5f} pp)")
    archive = output / "vadclip_v16_results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != archive and "__pycache__" not in path.parts:
                z.write(path, path.relative_to(output))
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Output ZIP integrity failed")
    print("Download:", archive)
    if any(r["status"] != "ok" for r in runs.values()):
        raise RuntimeError("Some runs failed. ZIP contains partial reports and logs, not a complete benchmark.")
    return str(archive)


RESULT_ZIP = run_both()
