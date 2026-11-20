"""Kaggle orchestrator; sources/config are embedded by the standalone notebook."""

import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

V0_CHECKPOINT_SHA = {
    "ucf": "d064aba05190ce5f6652151e6259b24ca6dda83a1e6289bf343afb6f905a2c79",
    "xd": "a676c14d794a7046f7636041e601a5d9a2bef03708dbc4cae90f8b37ac2af2b4",
}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_checkpoint(prefix, override, directory):
    candidates = [Path(override)] if override else sorted(
        p for p in INPUT.rglob("*") if p.is_file() and (
            p.name.lower() in {f"model_{prefix}.pth", "model.pth"}
            or p.suffix.lower() == ".zip" and prefix in p.stem.lower()
            and "checkpoint" in p.stem.lower()))
    for path in candidates:
        if path.suffix.lower() == ".pth" and file_sha(path) == V0_CHECKPOINT_SHA[prefix]:
            return path
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir() or not member.filename.lower().endswith(".pth"):
                        continue
                    with archive.open(member) as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest == V0_CHECKPOINT_SHA[prefix]:
                        directory.mkdir(parents=True, exist_ok=True)
                        target = directory / f"model_{prefix}.pth"
                        target.write_bytes(archive.read(member))
                        if file_sha(target) != digest:
                            raise ValueError("Staged V0 checkpoint hash mismatch")
                        return target
    raise FileNotFoundError(f"Attach the original pinned {prefix} V0 checkpoint (.pth or ZIP)")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runner_header(config):
    return "from pathlib import Path\n" + "\n".join(
        f"{name} = " + (f"Path({str(value)!r})" if isinstance(value, Path) else repr(value))
        for name, value in config.items()) + "\n\n"


def run_dataset(name, source, config, scripts):
    reporting = load_module("v21_reporting", scripts / "benchmark_reporting.py")
    reporting.save_run_config(scripts / f"{name}.config.json", config, source)
    script = scripts / f"{name}.py"
    script.write_text(runner_header(config) + source, encoding="utf-8")
    print(f"\n--- {name} ---", flush=True)
    last_lines = deque(maxlen=40)
    with (scripts.parent / f"{name}_console.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen([sys.executable, "-u", str(script)], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"}) as process:
            for line in process.stdout:
                last_lines.append(line)
                log.write(line)
                print(line, end="", flush=True)
            if process.wait() != 0:
                detail = "".join(last_lines).strip() or "Subprocess produced no console output"
                raise RuntimeError(f"Failed {name}; see {log.name}\nLast output:\n{detail}")
    paths = list(Path(config["OUTPUT"]).rglob("run.json"))
    for path in paths:
        reporting.emit_report(path)
        geometry = json.loads(path.read_text(encoding="utf-8")).get("geometry")
        if geometry:
            print(reporting.render_table(["Geometry setting", "Value"], [
                [key, geometry[key]] for key in ("space", "curvature_magnitude", "lambda_visual",
                                                "lambda_text", "temperature", "chunk_size")
            ]), flush=True)
    if config.get("CPU_PREFLIGHT"):
        return Path(config["OUTPUT"]) / "preflight.json"
    if source == TRAINER_SOURCE:
        return Path(config["OUTPUT"]) / "training.json"
    if len(paths) != 1:
        raise RuntimeError(f"Expected one evaluator run.json for {name}")
    return paths[0]


def verify_comparison(prefix, paths, analysis, treatment="v21"):
    import numpy as np

    reports, arrays = {}, {}
    for name, path in paths.items():
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_name("predictions.npz"), allow_pickle=False) as archive:
            arrays[name] = dict(archive)
    base_report, base = reports["v0"], arrays["v0"]
    primary = "classification" if prefix == "ucf" else "branch_2"
    metric = "roc_auc" if prefix == "ucf" else "frame_ap"
    expected = .8801992605166816 if prefix == "ucf" else .8450448809004643
    if abs(base_report["metrics"][primary][metric] - expected) > .00005:
        raise ValueError("V0 does not reproduce the locked reference")
    id_key = "video_id" if prefix == "ucf" else "official_path"
    for name, array in arrays.items():
        report = reports[name]
        for key in ("initial_v0_checkpoint_sha256", "split_sha256", "ground_truth_sha256",
                    "visual_block_length", "frames_per_feature"):
            if report[key] != base_report[key]:
                raise ValueError(f"Mismatched V0/variant protocol: {name}/{key}")
        for key in (id_key, "snippet_offsets", "score_columns", "frame_gt", "video_label"):
            if not np.array_equal(array[key], base[key]):
                raise ValueError(f"Mismatched evaluator grid: {name}/{key}")
        if array["scores"].shape != base["scores"].shape or not np.isfinite(array["scores"]).all():
            raise ValueError("Invalid alignment scores")
    rows = []
    for column, branch in enumerate(("classification", "alignment") if prefix == "ucf"
                                    else ("branch_1", "branch_2")):
        control = "v0"
        bootstrap = analysis.paired_bootstrap(
            arrays[control]["scores"][:, column], arrays[treatment]["scores"][:, column],
            base["frame_gt"].reshape(-1), base["snippet_offsets"], BOOTSTRAP_REPLICATES, SEED)
        for name, boot_key in (("roc_auc", "auc_ci95_pp"),
                               ("ap" if prefix == "ucf" else "frame_ap", "ap_ci95_pp")):
            a, b = [reports[v]["metrics"][branch][name] * 100 for v in (control, treatment)]
            low, high = bootstrap[boot_key] or (None, None)
            rows.append({"dataset": prefix, "branch": branch, "metric": name,
                         "control": control, "treatment": treatment, "control_pct": a,
                         "treatment_pct": b, "delta_pp": b - a,
                         "ci95_low_pp": low, "ci95_high_pp": high})
    return rows


def compare_historical(prefix, path, reference):
    """Compare saved ZIP5 metrics only after checking the same evaluator protocol."""
    report = json.loads(path.read_text(encoding="utf-8"))
    previous = reference["datasets"][prefix]
    for key, expected in previous["protocol"].items():
        if report[key] != expected:
            raise ValueError(f"Historical evaluator protocol mismatch: {prefix}/{key}")
    rows = []
    for branch, metrics in previous["metrics"].items():
        for name, value in metrics.items():
            if name not in {"roc_auc", "ap", "frame_ap", "ano_auc"}:
                continue
            current = report["metrics"][branch][name]
            rows.append({"dataset": prefix, "branch": branch, "metric": name,
                         "reference_run": reference["run"], "reference_variant": "v21_zip5",
                         "treatment": "cosine_control", "reference_pct": 100 * value,
                         "treatment_pct": 100 * current, "delta_pp": 100 * (current - value),
                         "scope": reference["scope"]})
    return rows


def dataset_train_config(config, prefix):
    resolved = dict(config)
    resolved.update(config.get("recipe_by_dataset", {}).get(prefix, {}))
    resolved["dataset"] = prefix
    rates = config.get("learning_rate_by_dataset")
    rate = rates[prefix] if rates is not None else config["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid learning rate for {prefix}: {rate}")
    resolved["learning_rate"] = float(rate)
    return resolved


def run_both():
    # Rerunning a failed notebook cell must not offer a stale ZIP from an earlier run.
    globals().pop("RESULT_ZIP", None)
    globals().pop("LAST_RESULT_ZIP", None)
    if RUN_MODE not in {"smoke", "pilot", "full"} or not INPUT.is_dir():
        raise ValueError("Use RUN_MODE smoke/pilot/full and attach Kaggle inputs")
    if globals().get("RECOVERY_FROM") and RUN_MODE != "full":
        raise ValueError("RECOVERY_FROM reuses completed full datasets; use RUN_MODE='full'")
    training_config = dict(TRAIN_CONFIG)
    native = bool(TRAIN_CONFIG.get("native_cosine_control", False))
    treatment = "cosine_control" if native else "v21"
    if RUN_MODE == "pilot":
        if not (native or TRAIN_CONFIG.get("sanity_fix")):
            raise ValueError("Train-only pilot is supported by the Sanity Fix recipe")
        pilot_epochs = TRAIN_CONFIG.get("pilot_epochs", 3)
        if type(pilot_epochs) is not int or pilot_epochs < 1:
            raise ValueError("pilot_epochs must be a positive integer")
        training_config["epochs"] = pilot_epochs
    if not DATASETS or len(set(DATASETS)) != len(DATASETS) or set(DATASETS) - {"ucf", "xd"}:
        raise ValueError("DATASETS must contain ucf and/or xd without duplicates")
    if BOOTSTRAP_REPLICATES < 1 or HALF_BATCH < 1 or TRAIN_CONFIG["epochs"] < 1:
        raise ValueError("Invalid training/analysis budget")
    root = WORK / "vadclip_v21"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run_", dir=root))
    scripts = output / "runtime_scripts"
    scripts.mkdir()
    for name, source in SOURCES.items():
        (scripts / name).write_text(source, encoding="utf-8")
    reporting = load_module("v21_reporting", scripts / "benchmark_reporting.py")
    analysis = load_module("v21_analysis", scripts / "analysis.py")
    latency = load_module("v21_latency", scripts / "latency_inference.py")
    experiment = {"version": "v21", "mode": RUN_MODE, "seed": SEED,
                  "geometry_config": GEOMETRY_CONFIG, "train_config": training_config,
                  "bootstrap": BOOTSTRAP_REPLICATES,
                  "training_recipe": ("COSINE_PINNED_REPOSITORY" if TRAIN_CONFIG.get("native_repository_recipe") else
                                      "COSINE_NATIVE_LR_CONTROL" if native else
                                      "SANITY_FIXED_C" if TRAIN_CONFIG.get("sanity_fix") else
                                      "PAPER" if TRAIN_CONFIG.get("paper_design") else
                                      "B" if "from_scratch" in TRAIN_CONFIG else "A"),
                  "reproduction_disclosure": TRAIN_CONFIG.get("reproduction_disclosure"),
                  "claim": ("pinned repo defaults/crop exposure; explicit RNG/audit differences; not an optimizer-only test"
                            if TRAIN_CONFIG.get("native_repository_recipe") else
                            "native cosine run_yxfrilug recipe with only LR changed; observational traces added; pilot has its own schedule"
                            if native else "disclosed sanity corrections; no author-exact or quality claim"
                            if TRAIN_CONFIG.get("sanity_fix") else
                            "published equations/settings; author-code parity unresolved"
                            if TRAIN_CONFIG.get("paper_design") else
                            "fresh downstream Poincare training; frozen CLIP only from V0"
                            if "from_scratch" in TRAIN_CONFIG else
                            "two-stage A Poincare adaptation, warm-start V0; CLIP frozen"),
                  "cosine_control_trained": False,
                  "trained_variant": treatment,
                  "full_poinclip_reproduction": False, "open_vocabulary": False,
                  "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest()
                                    for name, source in SOURCES.items()}}
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    shared = {"INPUT": INPUT, "WORK": WORK, "CLIP_WEIGHTS": CLIP_WEIGHTS,
              "SRC_REPO": Path(tempfile.gettempdir()) / "vadclip_xd_c41067f" / "upstream",
              "SMOKE_ONLY": RUN_MODE == "smoke", "LATENCY_WARMUP": LATENCY_WARMUP,
              "LATENCY_REPEATS": LATENCY_REPEATS, "GEOMETRY_CONFIG": GEOMETRY_CONFIG}
    jobs = {
        "ucf": (UCF_RUNNER, UCF_TRAIN_ROOT, UCF_CHECKPOINT, {"DATA_ROOT": UCF_DATA_ROOT}),
        "xd": (XD_RUNNER, XD_TRAIN_ROOT, XD_CHECKPOINT,
               {"DATA_ROOT": XD_DATA_ROOT, "FEATURE_ZIP": XD_FEATURE_ZIP,
                "REFERENCE_MANIFEST": None, "REQUIRE_PACK_MANIFEST": True, "RUN_MODE": RUN_MODE}),
    }
    results, comparisons, historical_rows = {}, [], []
    if native and not TRAIN_CONFIG.get("native_repository_recipe") and globals().get("ZIP5_REFERENCE"):
        (output / "historical_reference.json").write_text(
            json.dumps(ZIP5_REFERENCE, indent=2), encoding="utf-8")
    print("Output:", output, flush=True)
    crop_preflight_failures = {}
    if TRAIN_CONFIG.get("native_repository_recipe"):
        print("Validating ALL train inputs on CPU before any GPU evaluation/training.", flush=True)
        for prefix in DATASETS:
            try:
                run_dataset(f"{prefix}_crop_preflight",
                            "from v21_recipe import preflight_main\npreflight_main(globals())\n", {
                    **shared, "OUTPUT": output / prefix / "crop_preflight", "PREFIX": prefix,
                    "TRAIN_ROOT": jobs[prefix][1], "TRAIN_CONFIG": dataset_train_config(training_config, prefix),
                    "SEED": SEED, "CPU_PREFLIGHT": True, "RUN_MODE": RUN_MODE,
                }, scripts)
            except Exception as exc:
                crop_preflight_failures[prefix] = str(exc)
        if crop_preflight_failures:
            print("CPU input validation failed; NO GPU jobs will run for any dataset.", flush=True)
    for prefix in DATASETS:
        runner, train_root, override, inputs = jobs[prefix]
        paths = {}
        entry = results[prefix] = {"status": "running", "reports": {}}
        try:
            if crop_preflight_failures:
                raise RuntimeError("CPU crop validation failed: " + json.dumps(crop_preflight_failures))
            resolved_train_config = dataset_train_config(training_config, prefix)
            entry["resolved_learning_rate"] = resolved_train_config["learning_rate"]
            checkpoint = resolve_checkpoint(prefix, override, output / prefix / "v0_input")
            evaluation = {**shared, **inputs, "CHECKPOINT": checkpoint}
            reused = None
            if globals().get("RECOVERY_FROM"):
                recovery = load_module("v21_recovery", scripts / "v21_recovery.py")
                with recovery.RecoveryInput(RECOVERY_FROM) as old_run:
                    reused = old_run.reuse(prefix, output, SOURCES, resolved_train_config,
                                           GEOMETRY_CONFIG, SEED, V0_CHECKPOINT_SHA[prefix],
                                           TRAIN_MANIFESTS[prefix], treatment)
                if reused is not None:
                    paths, trained_path = reused
                    entry["reports"] = {name: path.relative_to(output).as_posix() for name,path in paths.items()}
                    entry["reused_completed_dataset"] = True
                    entry["recovery_receipt"] = f"{prefix}/recovery_receipt.json"
                    print(f"{prefix}: reusing completed training and evaluation; rerunning diagnostics.", flush=True)

            def evaluate(name, space=None, weights=None, expected_hash=None):
                path = run_dataset(f"{prefix}_{name}", runner, {
                    **evaluation, "OUTPUT": output / prefix / name, "VARIANT": name,
                    "V21_SPACE": space, "V21_MODEL_PATH": weights,
                    "V21_MODEL_SHA256": expected_hash,
                }, scripts)
                paths[name] = path
                entry["reports"][name] = path.relative_to(output).as_posix()

            if RUN_MODE != "pilot" and reused is None:
                evaluate("v0")
            if RUN_MODE == "full":
                baseline = json.loads(paths["v0"].read_text(encoding="utf-8"))
                branch, metric, expected = (("classification", "roc_auc", .8801992605166816)
                                            if prefix == "ucf" else
                                            ("branch_2", "frame_ap", .8450448809004643))
                if abs(baseline["metrics"][branch][metric] - expected) > .00005:
                    raise ValueError("V0 reference mismatch; training has not been started")
            train_output = output / prefix / "train"
            if reused is None:
                trained_path = run_dataset(f"{prefix}_train", TRAINER_SOURCE, {
                    **shared, "OUTPUT": train_output, "PREFIX": prefix, "TRAIN_ROOT": train_root,
                    "V0_PATH": checkpoint, "V0_SHA256": V0_CHECKPOINT_SHA[prefix],
                    "MANIFEST_SHA256": TRAIN_MANIFESTS[prefix], "SEED": SEED,
                    "CROP_INVENTORY": output / prefix / "crop_preflight" / "validated_crop_inventory.json",
                    "HALF_BATCH": HALF_BATCH, "TRAIN_CONFIG": resolved_train_config,
                    "RUN_MODE": RUN_MODE,
                }, scripts)
            trained = json.loads(trained_path.read_text(encoding="utf-8"))
            entry["training"] = trained_path.relative_to(output).as_posix()
            if native and not TRAIN_CONFIG.get("native_repository_recipe") and RUN_MODE == "full" and globals().get("ZIP5_REFERENCE"):
                for key, expected in ZIP5_REFERENCE["datasets"][prefix]["training"].items():
                    if trained.get(key) != expected:
                        raise ValueError(f"Historical training input mismatch: {prefix}/{key}")
            entry["cosine_control_trained"] = native
            if native:
                experiment["cosine_control_trained"] = all(
                    results.get(ds, {}).get("cosine_control_trained", False) for ds in DATASETS)
            for name in (treatment,):
                record = trained["variants"][name]
                weights = train_output / record["checkpoint_file"]
                if RUN_MODE != "pilot" and reused is None:
                    evaluate(name, record["space"], weights, record["checkpoint_sha256"])
            if RUN_MODE == "full":
                comparisons.extend(verify_comparison(prefix, paths, analysis, treatment)
                                   if native else verify_comparison(prefix, paths, analysis))
                if native and not TRAIN_CONFIG.get("native_repository_recipe") and globals().get("ZIP5_REFERENCE"):
                    historical_rows.extend(compare_historical(prefix, paths[treatment], ZIP5_REFERENCE))
                diagnostic = load_module("v21_error_analysis", scripts / "v21_error_analysis.py")
                diagnosis = diagnostic.analyze_benchmark(prefix, paths, output / prefix / "error_analysis")
                entry["error_analysis"] = f"{prefix}/error_analysis/error_summary.json"
                entry["error_segments"] = diagnosis["error_segment_count"]
            entry["status"] = "ok"
        except Exception as exc:  # Preserve partial outputs; never treat failure as success.
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(prefix, entry["error"], flush=True)
        finally:
            (output / "summary.json").write_text(json.dumps({
                **experiment, "runs": results, "branch_comparisons": comparisons,
            }, indent=2), encoding="utf-8")
        if entry["status"] == "failed" and STOP_ON_FAILURE:
            print("Stopping after failed dataset; saving partial ZIP.", flush=True)
            break
    postprocessing_errors = []
    collectors = (() if RUN_MODE == "pilot" else
                  (("reporting", reporting.collect_reports), ("latency", latency.collect_latency)))
    if RUN_MODE == "pilot":
        print("Train-only pilot complete; official test/latency not evaluated. Inspect learning_health.json.",
              flush=True)
    for name, collect in collectors:
        try:
            collect(output)
        except Exception as exc:
            postprocessing_errors.append({"step": name, "error": f"{type(exc).__name__}: {exc}"})
            print("Postprocessing failed:", postprocessing_errors[-1], flush=True)
    (output / "summary.json").write_text(json.dumps({
        **experiment, "runs": results, "branch_comparisons": comparisons,
        "postprocessing_errors": postprocessing_errors,
    }, indent=2), encoding="utf-8")
    if comparisons:
        with (output / "branch_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        print(reporting.render_table(
            ["Dataset", "Branch", "Metric", "Control", "Treatment", "Delta pp", "CI95 pp"],
            [[r["dataset"], r["branch"], r["metric"], r["control"], r["treatment"], f"{r['delta_pp']:+.5f}",
              ("N/A" if r["ci95_low_pp"] is None else
               f"[{r['ci95_low_pp']:+.5f}, {r['ci95_high_pp']:+.5f}]")]
             for r in comparisons],
            numeric=(5, 6)), flush=True)
    if historical_rows:
        with (output / "historical_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(historical_rows[0]))
            writer.writeheader()
            writer.writerows(historical_rows)
        print("Historical ZIP5 comparison saved: historical_comparison.csv; heads/objective differ.",
              flush=True)
    archive_path = output / "vadclip_v21_results.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and (path.suffix in {".json", ".jsonl", ".csv", ".py", ".log", ".md"}
                                   or path.name == "best.pth"
                                   or globals().get("INCLUDE_EPOCH_CHECKPOINTS", False) and path.name.startswith("epoch_") and path.suffix == ".pth"
                                   or path.name == "diagnostic_tensors.npz"
                                   or INCLUDE_PREDICTIONS and path.name == "predictions.npz"):
                archive.write(path, path.relative_to(output))
    globals()["LAST_RESULT_ZIP"] = archive_path
    print("Download:", archive_path, flush=True)
    if postprocessing_errors or any(result["status"] != "ok" for result in results.values()):
        reasons = [f"{dataset}: {record.get('error', 'incomplete')}"
                   for dataset, record in results.items() if record["status"] != "ok"]
        reasons.extend(f"{item['step']}: {item['error']}" for item in postprocessing_errors)
        raise RuntimeError(f"Incomplete V21 run. Partial results saved in {archive_path}\n"
                           + "\n".join(reasons))
    return archive_path


# Notebook imports the embedded sources/config before this cell.
# ruff: noqa: F821
RESULT_ZIP = run_both()
