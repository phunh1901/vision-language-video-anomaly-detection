"""Console and JSON/CSV reports for the existing closed-set WSVAD protocol."""

import csv
import hashlib
import json
from pathlib import Path

V0_REFERENCE = {
    "ucf": {"roc_auc": 0.8801992605166816, "ap": 0.3355725526808031,
            "ano_auc": 0.7023225660564671},
    "xd": {"roc_auc": 0.946812650099563, "ap": 0.8450448809004643},
}


def count_parameters(model, bundle=None):
    """Count resident model parameters, not FLOPs or feature extraction operations."""
    base = sum(parameter.numel() for parameter in model.parameters())
    extra = sum(parameter.numel() for parameter in bundle[0].parameters()) if bundle else 0
    return {"base_model": base, "additional_head": extra, "total_resident": base + extra,
            "scope": "loaded model including frozen encoders; plus separate head when enabled"}


def format_number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_table(headers, rows, numeric=()):
    """Use fixed character widths for notebook monospace console output."""
    headers = [str(value) for value in headers]
    cells = [["N/A" if value is None else str(value) for value in row] for row in rows]
    if any(len(row) != len(headers) for row in cells):
        raise ValueError("Table row does not match the number of columns")
    widths = [max(len(header), *(len(row[index]) for row in cells))
              if cells else len(header) for index, header in enumerate(headers)]

    def line(row, heading=False):
        return "| " + " | ".join(
            cell.rjust(widths[index]) if index in numeric and not heading
            else cell.ljust(widths[index]) for index, cell in enumerate(row)
        ) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    return "\n".join([separator, line(headers, True), separator,
                      *(line(row) for row in cells), separator])


def flatten_report(report, source):
    latency = report.get("latency") or {}
    dataset = latency.get("dataset")
    if dataset not in V0_REFERENCE:
        raise ValueError(f"Unknown dataset in evaluator report: {source}")
    primary = "classification" if dataset == "ucf" else "branch_2"
    values = (report.get("metrics") or {}).get(primary) or {}
    auc = values.get("roc_auc")
    ap = values.get("ap", values.get("frame_ap"))
    ano = values.get("ano_auc")
    params = report.get("parameters") or {}
    return {
        "dataset": dataset, "variant": latency.get("variant"), "mode": latency.get("mode"),
        "primary_branch": primary, "video_count": latency.get("video_count"),
        "roc_auc_pct": None if auc is None else 100 * auc,
        "ap_pct": None if ap is None else 100 * ap,
        "ano_auc_pct": None if ano is None else 100 * ano,
        "auc_vs_reference_v0_pp": None if auc is None else
            100 * (auc - V0_REFERENCE[dataset]["roc_auc"]),
        "ap_vs_reference_v0_pp": None if ap is None else
            100 * (ap - V0_REFERENCE[dataset]["ap"]),
        "mean_ms_per_video": latency.get("mean_ms_per_video"),
        "median_ms_per_video": latency.get("median_ms_per_video"),
        "p95_ms_per_video": latency.get("p95_ms_per_video"),
        "ms_per_snippet": latency.get("ms_per_snippet"),
        "snippets_per_second": latency.get("snippets_per_second"),
        "peak_torch_vram_mib": report.get("peak_torch_vram_mib"),
        "model_parameters": params.get("base_model"),
        "additional_head_parameters": params.get("additional_head"),
        "total_resident_parameters": params.get("total_resident"),
        "gpu": latency.get("gpu"), "input_dtypes": ", ".join(latency.get("input_dtypes", [])),
        "warmup": latency.get("warmup_calls_first_video"),
        "repeats": latency.get("repeats_per_video"),
        "visual_block_length": report.get("visual_block_length"),
        "frames_per_feature": report.get("frames_per_feature"),
        "torch": latency.get("torch"), "cuda_runtime": latency.get("cuda_runtime"),
        "checkpoint_sha256": report.get("evaluated_checkpoint_sha256"),
        "report_path": str(source),
    }


def emit_report(path):
    """Print after the evaluator has successfully saved its report."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    row = flatten_report(report, path)
    print(f"\n=== RESULTS {row['dataset'].upper()} / {row['variant']} / {row['mode']} ===",
          flush=True)
    print(f"Primary branch: {row['primary_branch']}; videos: {row['video_count']}", flush=True)
    if report.get("metrics"):
        branch_rows = []
        for name, values in report["metrics"].items():
            numbers = [values.get("roc_auc"), values.get("ap", values.get("frame_ap")),
                       values.get("ano_auc")]
            branch_rows.append([name, "primary" if name == row["primary_branch"] else "auxiliary",
                                *(format_number(None if n is None else n * 100) for n in numbers)])
        print(render_table(["Branch", "Role", "ROC-AUC (%)", "AP (%)", "Ano-AUC (%)"],
                           branch_rows, numeric=(2, 3, 4)), flush=True)
        print("Historical reproduced V0 reference check (pp):", flush=True)
        print(render_table(["Metric", "Difference"], [
            ["ROC-AUC", format_number(row["auc_vs_reference_v0_pp"], 5)],
            ["AP", format_number(row["ap_vs_reference_v0_pp"], 5)],
        ], numeric=(1,)), flush=True)
    else:
        print("Benchmark: N/A (smoke test; not a full evaluation)", flush=True)
    print("Latency feature -> logits and memory:", flush=True)
    print(render_table(["Measurement", "Value", "Unit"], [
        ["Mean", format_number(row["mean_ms_per_video"], 3), "ms/video"],
        ["Median", format_number(row["median_ms_per_video"], 3), "ms/video"],
        ["P95", format_number(row["p95_ms_per_video"], 3), "ms/video"],
        ["Amortized latency", format_number(row["ms_per_snippet"], 6), "ms/snippet"],
        ["Throughput", format_number(row["snippets_per_second"], 2), "snippet/s"],
        ["Peak allocated VRAM", format_number(row["peak_torch_vram_mib"], 2), "MiB"],
    ], numeric=(1,)), flush=True)
    parameter_rows = []
    for label, key in (("Loaded model", "model_parameters"),
                       ("Additional head", "additional_head_parameters"),
                       ("Total resident", "total_resident_parameters")):
        count = row[key]
        parameter_rows.append([label, "N/A" if count is None else f"{count:,}",
                               "N/A" if count is None else f"{count / 1e6:.4f}"])
    print(render_table(["Parameters", "Count", "Millions"], parameter_rows,
                       numeric=(1, 2)), flush=True)
    print(render_table(["Run setting", "Value"], [
        [label, row[key]] for label, key in (
            ("GPU", "gpu"), ("Input dtype", "input_dtypes"),
            ("Block length", "visual_block_length"), ("Frames/snippet", "frames_per_feature"),
            ("Warm-up calls", "warmup"), ("Repeats/video", "repeats"),
            ("Torch", "torch"), ("CUDA runtime", "cuda_runtime"),
        )
    ]), flush=True)
    print("Offline feature inference; excludes raw video decode and visual CLIP extraction. "
          "No Base/Novel or online end-to-end metrics in this protocol.", flush=True)
    print(f"Report: {path}\n", flush=True)
    return row


def collect_reports(output):
    output = Path(output)
    paths = sorted(output.rglob("run.json"))
    rows = []
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if "latency" not in report:
            continue
        rows.append(flatten_report(report, path.relative_to(output).as_posix()))
        reports.append({"path": path.relative_to(output).as_posix(), "report": report})
    if not rows:
        return []
    baseline = {(r["dataset"], r["mode"]): r for r in rows if r["variant"] == "v0"}
    for row in rows:
        ref = baseline.get((row["dataset"], row["mode"]))
        for key, out in (("roc_auc_pct", "auc_vs_same_run_v0_pp"),
                         ("ap_pct", "ap_vs_same_run_v0_pp"),
                         ("ano_auc_pct", "ano_auc_vs_same_run_v0_pp")):
            row[out] = (row[key] - ref[key] if ref and ref.get(key) is not None
                        and row.get(key) is not None else None)
        for key, out in (("mean_ms_per_video", "mean_latency_vs_same_run_v0_pct"),
                         ("ms_per_snippet", "snippet_latency_vs_same_run_v0_pct")):
            row[out] = (100 * (row[key] / ref[key] - 1) if ref and ref.get(key)
                        and row.get(key) is not None else None)
    with (output / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "benchmark_summary.json").write_text(json.dumps({
        "protocol": "closed-set WSVAD; no Base/Novel, no FLOPs",
        "metrics_scale": "percent; differences in percentage points",
        "rows": rows, "reports": reports,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY: primary metrics and latency ===", flush=True)
    print("QUALITY: differences use V0 from the same dataset/mode/run", flush=True)
    print(render_table(
        ["Dataset", "Variant", "AUC (%)", "AP (%)", "Ano-AUC (%)", "AUC diff pp", "AP diff pp"],
        [[r["dataset"], r["variant"], *(format_number(r[key]) for key in (
            "roc_auc_pct", "ap_pct", "ano_auc_pct", "auc_vs_same_run_v0_pp",
            "ap_vs_same_run_v0_pp"))] for r in rows], numeric=(2, 3, 4, 5, 6)), flush=True)
    print("LATENCY: ms/video; difference uses same-run V0", flush=True)
    print(render_table(
        ["Dataset", "Variant", "Mean", "Median", "P95", "ms/snippet", "Mean diff (%)"],
        [[r["dataset"], r["variant"], *(format_number(r[key], 5) for key in (
            "mean_ms_per_video", "median_ms_per_video", "p95_ms_per_video", "ms_per_snippet",
            "mean_latency_vs_same_run_v0_pct"))] for r in rows], numeric=(2, 3, 4, 5, 6)), flush=True)
    print("Saved benchmark_summary.csv and benchmark_summary.json", flush=True)
    return rows


def save_run_config(path, config, runner):
    """Persist exact subprocess configuration before it starts, including failed runs."""
    path = Path(path)
    receipt = {"config": config, "runner_sha256": hashlib.sha256(runner.encode()).hexdigest()}
    path.write_text(json.dumps(receipt, default=str, indent=2, ensure_ascii=False), encoding="utf-8")
