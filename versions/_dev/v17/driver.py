# ruff: noqa: F821
"""Notebook driver. Earlier cells supply configuration and embedded source."""

VARIANTS = ("v0", "v7", "residual_only", "v17")
PAIRS = (("v0", "v7"), ("v0", "residual_only"), ("v7", "v17"), ("v0", "v17"))
BASELINE = {"ucf": ("classification", "roc_auc", 0.8801992605166816),
            "xd": ("branch_2", "frame_ap", 0.8450448809004643)}


def compare(prefix, paths, analysis):
    import numpy as np

    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    reference = reports["v0"]
    count = {"ucf": 290, "xd": 800}[prefix]
    hash_keys = ("split_sha256", "ground_truth_sha256", "evaluated_checkpoint_sha256",
                 "visual_block_length", "frames_per_feature")
    for report in reports.values():
        if report["run_kind"] != "full_checkpoint_inference" or report["video_count"] != count:
            raise ValueError("Full inference report missing official videos.")
        if any(report[key] != reference[key] for key in hash_keys):
            raise ValueError("Split, checkpoint or frame grid mismatch.")
    branch, metric, target = BASELINE[prefix]
    if not abs(reference["metrics"][branch][metric] - target) <= 0.00005:
        raise ValueError("V0 reproduction check failed.")

    arrays = {}
    for key, path in paths.items():
        with np.load(path.with_name("predictions.npz"), allow_pickle=False) as data:
            arrays[key] = dict(data)
    base = arrays["v0"]
    mapping = ("snippet_offsets", "frame_gt", "score_columns", "video_label",
               "video_id" if prefix == "ucf" else "official_path")
    for key, current in arrays.items():
        if any(not np.array_equal(base[k], current[k]) for k in mapping):
            raise ValueError(f"{prefix}/{key}: mismatched predictions mapping.")
        if current["scores"].shape != base["scores"].shape or not np.isfinite(current["scores"]).all():
            raise ValueError(f"{prefix}/{key}: invalid snippet scores.")
    for control, treatment in (("v0", "residual_only"), ("v7", "v17")):
        if not np.array_equal(arrays[control]["scores"][:, 0], arrays[treatment]["scores"][:, 0]):
            raise ValueError(f"{prefix}: C branch differs for {treatment} vs {control}.")
    offsets, gt = base["snippet_offsets"], base["frame_gt"].reshape(-1)
    if reference["frames_per_feature"] != 16 or len(gt) != int(offsets[-1]) * 16:
        raise ValueError("Expected 16 frames per snippet.")

    rows = []
    summary = {}
    names = ("classification", "alignment") if prefix == "ucf" else ("branch_1", "branch_2")
    for control, treatment in PAIRS:
        label = f"{treatment}_vs_{control}"
        for column, name in enumerate(names):
            c, t = (arrays[v]["scores"][:, column] for v in (control, treatment))
            bootstrap = analysis.paired_bootstrap(c, t, gt, offsets, BOOTSTRAP_REPLICATES, SEED)
            for native_metric, bounds_key in (
                ("roc_auc", "auc_ci95_pp"),
                ("ap" if prefix == "ucf" else "frame_ap", "ap_ci95_pp"),
                *((("ano_auc", None),) if prefix == "ucf" else ()),
            ):
                a, b = (100 * reports[v]["metrics"][name][native_metric] for v in (control, treatment))
                bounds = (bootstrap.get(bounds_key) or [None, None])
                row = {
                    "dataset": prefix, "comparison": label, "control": control,
                    "treatment": treatment, "branch": "CA"[column], "metric": native_metric,
                    "primary": name == branch and native_metric == metric,
                    "control_pct": a, "treatment_pct": b, "delta_pp": b - a,
                    "ci95_low_pp": bounds[0], "ci95_high_pp": bounds[1],
                }
                rows.append(row)
                if row["primary"]:
                    summary[label] = row
    return rows, summary


def format_table(headers, values):
    widths = [max(map(len, column)) for column in zip(headers, *values, strict=True)]

    def line(cells):
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(cells, widths, strict=True)
        ) + " |"

    return "\n".join([line(headers), line(["-" * width for width in widths]),
                      *(line(row) for row in values)])


def comparison_report(rows):
    datasets = {"ucf": "UCF-Crime", "xd": "XD-Violence"}
    metrics = {"roc_auc": "ROC-AUC", "ap": "Frame AP", "frame_ap": "Frame AP",
               "ano_auc": "Ano-AUC"}
    variants = {"v0": "V0", "v7": "V7", "residual_only": "Residual riêng", "v17": "V17"}
    scores, differences = {}, []
    for row in rows:
        dataset, branch, metric = row["dataset"], row["branch"], row["metric"]
        label = f'{metrics[metric]}{" *" if row["primary"] else ""}'
        key = (datasets[dataset], branch, label)
        scores.setdefault(key, {}).update({
            row["control"]: row["control_pct"], row["treatment"]: row["treatment_pct"],
        })
        if row["primary"] or (row["comparison"] == "v17_vs_v7" and branch == "A"):
            low, high = row["ci95_low_pp"], row["ci95_high_pp"]
            bounds = "—" if low is None or high is None else f"[{low:+.5f}, {high:+.5f}]"
            pair = f'{variants[row["treatment"]]} − {variants[row["control"]]}'
            differences.append([*key, pair, f'{row["delta_pp"]:+.5f}', bounds])

    score_rows = [[*key, *(f"{values[v]:.5f}%" if v in values else "—" for v in VARIANTS)]
                  for key, values in scores.items()]
    return "\n\n".join([
        "## Điểm theo cấu hình",
        format_table(["Tập dữ liệu", "Nhánh", "Chỉ số", *variants.values()], score_rows),
        "## Chênh lệch so với đối chứng",
        format_table(["Tập dữ liệu", "Nhánh", "Chỉ số", "So sánh", "Δ (đpt)", "CI 95% (đpt)"],
                     differences),
        "* Chỉ số chính của tập dữ liệu. CI bootstrap ghép cặp theo video. "
        "Các đối chiếu đầy đủ nằm trong comparison.csv.",
    ])


def run_both():
    if RUN_MODE not in {"smoke", "full"}:
        raise ValueError("Chọn RUN_MODE = 'smoke' hoặc 'full'.")
    if not DATASETS or len(set(DATASETS)) != len(DATASETS) or set(DATASETS) - {"ucf", "xd"}:
        raise ValueError("DATASETS cần ucf, xd hoặc cả hai, không lặp.")
    if not INPUT.is_dir():
        raise FileNotFoundError("Gắn bốn Kaggle Dataset qua Add Input.")
    if not 0 <= RESIDUAL_ETA <= 1 or BOOTSTRAP_REPLICATES < 1:
        raise ValueError("Kiểm tra RESIDUAL_ETA và BOOTSTRAP_REPLICATES.")
    root = WORK / "vadclip_v17"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run_", dir=root))
    scripts = output / "runtime_scripts"
    scripts.mkdir()
    sources = {"v7_windows.py": WINDOW_SOURCE, "v17_residual.py": RESIDUAL_SOURCE,
               "analysis.py": ANALYSIS_SOURCE}
    for filename, contents in sources.items():
        (scripts / filename).write_text(contents, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("v17_analysis", scripts / "analysis.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    shared = {
        "INPUT": INPUT, "WORK": WORK, "CLIP_WEIGHTS": CLIP_WEIGHTS,
        "RESIDUAL_ETA": RESIDUAL_ETA, "SMOKE_ONLY": RUN_MODE == "smoke",
        "SRC_REPO": Path(tempfile.gettempdir()) / "vadclip_xd_c41067f" / "upstream",
    }
    jobs = {
        "ucf": (UCF_RUNNER, {"DATA_ROOT": UCF_DATA_ROOT, "CHECKPOINT": UCF_CHECKPOINT}),
        "xd": (XD_RUNNER, {
            "DATA_ROOT": XD_DATA_ROOT, "CHECKPOINT": XD_CHECKPOINT, "FEATURE_ZIP": XD_FEATURE_ZIP,
            "REFERENCE_MANIFEST": None, "REQUIRE_PACK_MANIFEST": True, "RUN_MODE": RUN_MODE,
        }),
    }
    started = time.perf_counter()
    runs, rows = {}, []
    experiment = {
        "mode": RUN_MODE, "eta": RESIDUAL_ETA, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED, "variants": VARIANTS, "include_predictions": INCLUDE_PREDICTIONS,
        "evaluation_status": "exploratory_on_previously_seen_official_test",
        "source_sha256": {key: hashlib.sha256(value.encode()).hexdigest()
                          for key, value in {**sources, "ucf": UCF_RUNNER, "xd": XD_RUNNER}.items()},
    }
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print("Output:", output, flush=True)
    for prefix in DATASETS:
        runner, config = jobs[prefix]
        paths = {}
        entry = runs[prefix] = {"status": "running", "reports": {}}
        try:
            for variant in VARIANTS:
                dest = output / prefix / variant
                run_dataset(f"{prefix}_{variant}", runner, {
                    **shared, **config, "VARIANT": variant, "OUTPUT": dest,
                }, scripts)
                reports = list((dest / RUN_MODE).rglob("run.json"))
                if len(reports) != 1:
                    raise ValueError(f"Expected one report for {prefix}/{variant}.")
                paths[variant] = reports[0]
                entry["reports"][variant] = reports[0].relative_to(output).as_posix()
            if RUN_MODE == "full":
                metrics, entry["primary"] = compare(prefix, paths, analysis)
                rows.extend(metrics)
            entry["status"] = "ok"
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"{prefix}: {entry['error']}", flush=True)
        finally:
            (output / "summary.json").write_text(json.dumps({
                **experiment, "runs": runs, "metrics": rows,
                "elapsed_seconds": time.perf_counter() - started,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "comparison.csv", rows)
    if rows:
        report = comparison_report(rows)
        (output / "comparison.md").write_text(report + "\n", encoding="utf-8")
        print("\n" + report, flush=True)
    archive = output / "vadclip_v17_results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".log", ".csv", ".py", ".md"}:
                z.write(path, path.relative_to(output))
            elif INCLUDE_PREDICTIONS and path.name == "predictions.npz":
                z.write(path, path.relative_to(output))
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Output ZIP integrity failed.")
    print("Download:", archive, flush=True)
    if any(entry["status"] != "ok" for entry in runs.values()):
        raise RuntimeError("Có lượt chưa hoàn tất. Xem summary.json và console log trong ZIP.")
    return archive


RESULT_ZIP = run_both()
