# ruff: noqa: F821
"""Kaggle notebook entry point. Earlier cells embed the evaluators and config."""

import time

VARIANTS = ("v7", "clip_only", "temporal_random", "v18")
PAIRS = (("v7", "clip_only"), ("clip_only", "temporal_random"),
         ("temporal_random", "v18"), ("v7", "v18"))


def compare_variants(prefix, paths, analysis):
    import numpy as np

    arrays, reports = {}, {}
    for name, path in paths.items():
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_name("predictions.npz"), allow_pickle=False) as data:
            arrays[name] = dict(data)
    baseline = reports["v7"]
    base = arrays["v7"]
    id_key = "video_id" if prefix == "ucf" else "official_path"
    offsets = base["snippet_offsets"]
    gt = base["frame_gt"].reshape(-1)
    expected = 290 if prefix == "ucf" else 800
    if len(gt) != offsets[-1] * 16 or len(offsets) != expected + 1:
        raise ValueError("Invalid official frame grid.")
    primary_column = 0 if prefix == "ucf" else 1
    primary_branch = "classification" if prefix == "ucf" else "branch_2"
    primary_metric = "roc_auc" if prefix == "ucf" else "frame_ap"
    target = 0.8802335993846885 if prefix == "ucf" else 0.8461979894100824
    if not abs(baseline["metrics"][primary_branch][primary_metric] - target) <= 0.00005:
        raise ValueError("V7 reproduction check failed.")
    for name, current in arrays.items():
        report = reports[name]
        keys = ("split_sha256", "ground_truth_sha256", "evaluated_checkpoint_sha256",
                "frames_per_feature", "visual_block_length")
        if (report["run_kind"] != "full_checkpoint_inference" or report["video_count"] != expected
                or any(report[k] != baseline[k] for k in keys)):
            raise ValueError(f"Wrong checkpoint or evaluation grid: {prefix}/{name}")
        if (current["scores"].shape != (offsets[-1], 2) or
                not np.isfinite(current["scores"]).all() or
                any(not np.array_equal(current[k], base[k]) for k in
                    (id_key, "snippet_offsets", "frame_gt", "score_columns", "video_label"))):
            raise ValueError(f"Mismatched evaluator outputs: {prefix}/{name}")
        if not np.array_equal(current["scores"][:, 1-primary_column],
                              base["scores"][:, 1-primary_column]):
            raise ValueError(f"Unexpected change to the secondary branch: {prefix}/{name}")

    rows = []
    names = ("classification", "alignment") if prefix == "ucf" else ("branch_1", "branch_2")
    for control, treatment in PAIRS:
        for col, branch in enumerate(names):
            c = arrays[control]["scores"][:, col]
            t = arrays[treatment]["scores"][:, col]
            boot = analysis.paired_bootstrap(c, t, gt, offsets, BOOTSTRAP_REPLICATES, SEED)
            for metric in reports[control]["metrics"][branch]:
                if metric not in {"roc_auc", "ap", "frame_ap", "ano_auc"}:
                    continue
                a, b = (reports[v]["metrics"][branch][metric] * 100 for v in (control, treatment))
                key = {"roc_auc": "auc_ci95_pp", "ap": "ap_ci95_pp", "frame_ap": "ap_ci95_pp"}
                low, high = boot.get(key.get(metric)) or (None, None)
                rows.append({
                    "dataset": prefix, "branch": "CA"[col], "metric": metric,
                    "primary": col == primary_column and metric == primary_metric,
                    "comparison": f"{treatment}_vs_{control}", "control": control,
                    "treatment": treatment, "control_pct": a, "treatment_pct": b, "delta_pp": b-a,
                    "ci95_low_pp": low, "ci95_high_pp": high,
                })
    coverage = analysis.compare_coverage(
        prefix, paths, reports, (0.01, 0.05), SEED, BOOTSTRAP_REPLICATES, "v7", "v18",
    )
    return rows, coverage


def report_path(destination):
    files = list((destination / RUN_MODE).rglob("run.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one evaluator report: {destination}")
    return files[0]


def run_both():
    import numpy as np

    if RUN_MODE not in {"smoke", "full"} or not DATASETS:
        raise ValueError("Use RUN_MODE='smoke' or 'full' and a nonempty DATASETS.")
    if len(set(DATASETS)) != len(DATASETS) or set(DATASETS) - {"ucf", "xd"}:
        raise ValueError("DATASETS accepts ucf and xd without duplicates.")
    if BOOTSTRAP_REPLICATES < 1 or not INPUT.is_dir():
        raise ValueError("Check BOOTSTRAP_REPLICATES and Kaggle Add Input.")
    root = WORK / "vadclip_v18"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run_", dir=root))
    scripts = output / "runtime_scripts"
    scripts.mkdir()
    for name, source in SOURCES.items():
        (scripts / name).write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("v18_analysis", scripts / "analysis.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    shared = {
        "INPUT": INPUT, "WORK": WORK, "CLIP_WEIGHTS": CLIP_WEIGHTS,
        "SMOKE_ONLY": RUN_MODE == "smoke",
        "SRC_REPO": Path(tempfile.gettempdir()) / "vadclip_xd_c41067f" / "upstream",
    }
    jobs = {
        "ucf": (UCF_RUNNER, UCF_TRAIN_ROOT, UCF_CHECKPOINT, {"DATA_ROOT": UCF_DATA_ROOT}),
        "xd": (XD_RUNNER, XD_TRAIN_ROOT, XD_CHECKPOINT, {
            "DATA_ROOT": XD_DATA_ROOT, "FEATURE_ZIP": XD_FEATURE_ZIP, "REFERENCE_MANIFEST": None,
            "REQUIRE_PACK_MANIFEST": True, "RUN_MODE": RUN_MODE,
        }),
    }
    started = time.perf_counter()
    results, rows = {}, []
    experiment = {
        "version": "v18", "mode": RUN_MODE, "seed": SEED, "bootstrap": BOOTSTRAP_REPLICATES,
        "training": TEMPORAL_CONFIG,
        "source_sha256": {name: hashlib.sha256(code.encode()).hexdigest() for name, code in
                          {**SOURCES, "trainer": TRAINER_SOURCE, "ucf": UCF_RUNNER, "xd": XD_RUNNER}.items()},
    }
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print("Output:", output, flush=True)
    for prefix in DATASETS:
        runner, train_root, checkpoint_override, config = jobs[prefix]
        entry = results[prefix] = {"status": "running", "reports": {}}
        train_output = output / prefix / "train"
        paths = {}
        try:
            checkpoint = resolve_checkpoint(prefix, checkpoint_override, output / prefix / "v0_input")
            evaluation = {**shared, **config, "CHECKPOINT": checkpoint}
            dest = output / prefix / "v7"
            run_dataset(f"{prefix}_v7", runner, {
                **evaluation, "OUTPUT": dest, "VARIANT": "v7",
                "HEAD_PATH": None, "STRENGTH": 0., "USE_TEMPORAL": False,
            }, scripts)
            paths["v7"] = report_path(dest)
            entry["reports"]["v7"] = paths["v7"].relative_to(output).as_posix()
            run_dataset(f"{prefix}_train", TRAINER_SOURCE, {
                **shared, "OUTPUT": train_output, "PREFIX": prefix, "TRAIN_ROOT": train_root,
                "V0_PATH": checkpoint, "V0_SHA256": V0_CHECKPOINT_SHA[prefix],
                "MANIFEST_SHA256": TRAIN_MANIFESTS[prefix], "SEED": SEED, "HALF_BATCH": HALF_BATCH,
                "VARIANTS": VARIANTS[1:], "TEMPORAL_CONFIG": TEMPORAL_CONFIG,
            }, scripts)
            trained = json.loads((train_output / "training.json").read_text(encoding="utf-8"))
            entry["training"] = (train_output / "training.json").relative_to(output).as_posix()
            for variant in VARIANTS[1:]:
                record = trained["variants"][variant]
                weights = train_output / f"{variant}.pth"
                if file_sha(weights) != record["checkpoint_sha256"]:
                    raise ValueError(f"Head checkpoint hash mismatch: {prefix}/{variant}")
                dest = output / prefix / variant
                run_dataset(f"{prefix}_{variant}", runner, {
                    **evaluation, "OUTPUT": dest, "VARIANT": variant, "HEAD_PATH": weights,
                    "STRENGTH": 1. if RUN_MODE == "smoke" else record["strength"],
                    "USE_TEMPORAL": variant != "clip_only",
                }, scripts)
                paths[variant] = report_path(dest)
                entry["reports"][variant] = paths[variant].relative_to(output).as_posix()
            if RUN_MODE == "full":
                metrics, entry["coverage_v18_vs_v7"] = compare_variants(prefix, paths, analysis)
                rows.extend(metrics)
                column = 0 if prefix == "ucf" else 1
                entry["false_alarm_at_recall"] = {}
                for variant in ("v7", "v18"):
                    with np.load(paths[variant].with_name("predictions.npz"), allow_pickle=False) as data:
                        entry["false_alarm_at_recall"][variant] = analysis.false_alarm_diagnostics(
                            data["scores"][:, column], data["frame_gt"].reshape(-1), data["snippet_offsets"],
                        )
            entry["status"] = "ok"
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(prefix, entry["error"], flush=True)
        finally:
            (output / "summary.json").write_text(json.dumps({
                **experiment, "runs": results, "metrics": rows,
                "elapsed_seconds": time.perf_counter() - started,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        report = comparison_report(rows)
        (output / "comparison.md").write_text(report + "\n", encoding="utf-8")
        print(report, flush=True)
    archive = output / "vadclip_v18_results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and (
                path.suffix in {".json", ".csv", ".md", ".log", ".py"}
                or path.suffix == ".pth" and path.parent.name == "train"
                or INCLUDE_PREDICTIONS and path.name == "predictions.npz"
            ):
                handle.write(path, path.relative_to(output))
    print("Download:", archive, flush=True)
    if any(entry["status"] != "ok" for entry in results.values()):
        raise RuntimeError("Lượt chạy thiếu kết quả. Xem summary.json và console log.")
    return archive


RESULT_ZIP = run_both()
