"""Check log/file agreement and real notebook subprocess reporting without a GPU."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"


ROOT = (REPOSITORY / 'versions/_dev')
SPEC = importlib.util.spec_from_file_location("reporting_test", SHARED / "benchmark_reporting.py")
REPORTING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTING)


def fixture_report(dataset="ucf", variant="v0", mode="full", latency=10.):
    metrics = ({"classification": {"roc_auc": .88, "ap": .33, "ano_auc": .70},
                "alignment": {"roc_auc": .85, "ap": .26, "ano_auc": .65}}
               if dataset == "ucf" else
               {"branch_1": {"roc_auc": .90, "frame_ap": .80},
                "branch_2": {"roc_auc": .94, "frame_ap": .84}})
    return {
        "metrics": metrics if mode == "full" else None,
        "parameters": {"base_model": 100, "additional_head": 0, "total_resident": 100},
        "peak_torch_vram_mib": 123., "visual_block_length": 256, "frames_per_feature": 16,
        "latency": {
            "dataset": dataset, "variant": variant, "mode": mode, "video_count": 3,
            "mean_ms_per_video": latency, "median_ms_per_video": latency,
            "p95_ms_per_video": latency + 1, "ms_per_snippet": latency / 100,
            "snippets_per_second": 100000 / latency, "input_dtypes": ["torch.float32"],
            "warmup_calls_first_video": 3, "repeats_per_video": 3, "gpu": "test GPU",
            "torch": "test", "cuda_runtime": "test",
        },
    }


def test_count_parameters_includes_frozen_and_extra_head():
    model = torch.nn.Linear(2, 3)
    model.requires_grad_(False)
    head = torch.nn.Linear(3, 1)
    assert REPORTING.count_parameters(model)["base_model"] == 9
    counts = REPORTING.count_parameters(model, (head, 1., True, True))
    assert counts["additional_head"] == 4
    assert counts["total_resident"] == 13


@pytest.mark.parametrize("dataset,auc,ap", [("ucf", 88., 33.), ("xd", 94., 84.)])
def test_primary_branch_and_console_match_report(tmp_path, capsys, dataset, auc, ap):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(fixture_report(dataset)), encoding="utf-8")
    row = REPORTING.emit_report(path)
    assert row["roc_auc_pct"] == pytest.approx(auc)
    assert row["ap_pct"] == pytest.approx(ap)
    output = capsys.readouterr().out
    assert any("P95" in line and "11.000" in line and "ms/video" in line
               for line in output.splitlines())
    assert any("Additional head" in line and "0.0000" in line for line in output.splitlines())
    assert "Primary branch: " + row["primary_branch"] in output
    assert not any("base_auc" in key or "novel" in key or "flops" in key for key in row)


def test_smoke_does_not_report_zero_accuracy(tmp_path, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(fixture_report(mode="smoke")), encoding="utf-8")
    row = REPORTING.emit_report(path)
    assert row["roc_auc_pct"] is None and row["ap_pct"] is None
    assert "Benchmark: N/A" in capsys.readouterr().out


def test_summary_uses_same_dataset_and_mode_baseline(tmp_path):
    examples = [fixture_report(), fixture_report(variant="v7", latency=12.),
                fixture_report(dataset="xd", variant="v7", latency=20.),
                fixture_report(variant="v7", mode="smoke", latency=8.)]
    for index, report in enumerate(examples):
        path = tmp_path / str(index) / "run.json"
        path.parent.mkdir()
        path.write_text(json.dumps(report), encoding="utf-8")
    rows = REPORTING.collect_reports(tmp_path)
    by = {(r["dataset"], r["variant"], r["mode"]): r for r in rows}
    assert by["ucf", "v7", "full"]["mean_latency_vs_same_run_v0_pct"] == pytest.approx(20.)
    assert by["xd", "v7", "full"]["mean_latency_vs_same_run_v0_pct"] is None
    assert by["ucf", "v7", "smoke"]["mean_latency_vs_same_run_v0_pct"] is None
    assert by["ucf", "v7", "full"]["auc_vs_same_run_v0_pp"] == 0
    assert by["xd", "v7", "full"]["auc_vs_same_run_v0_pp"] is None
    saved = json.loads((tmp_path / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert saved["rows"] == rows
    assert len(saved["reports"]) == 4
    assert (tmp_path / "benchmark_summary.csv").is_file()


@pytest.mark.parametrize("version", ["v0", "v7", "v20"])
def test_notebook_runs_subprocess_and_prints_saved_results(tmp_path, capsys, version):
    nb = json.loads((NOTEBOOKS / f"V{int(version.removeprefix('v')):02d}.ipynb").read_text(encoding="utf-8"))
    payload = {}
    exec("".join(nb["cells"][2 if version == "v20" else 3]["source"]), payload)
    source = (payload["SOURCES"]["benchmark_reporting.py"] if version == "v20"
              else payload["REPORT_SOURCE"])
    assert source == (SHARED / "benchmark_reporting.py").read_text(encoding="utf-8")
    scripts = tmp_path / "runtime_scripts"
    scripts.mkdir()
    (scripts / "benchmark_reporting.py").write_text(source, encoding="utf-8")
    tree = ast.parse("".join(nb["cells"][3 if version == "v20" else 8]["source"]))
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"run_dataset", "runner_header"}]
    namespace = {"Path": Path, "sys": sys, "os": os, "subprocess": subprocess}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook", "exec"), namespace)
    runner = ("import json\nOUTPUT.mkdir(parents=True, exist_ok=True)\n"
              f"report = {fixture_report()!r}\n"
              "(OUTPUT / 'run.json').write_text(json.dumps(report), encoding='utf-8')\n"
              "print('inference finished', flush=True)\n")
    namespace["run_dataset"]("ucf_v0", runner, {"OUTPUT": tmp_path / "output"}, scripts)
    log = capsys.readouterr().out
    assert "inference finished" in log
    assert "RESULTS UCF / v0 / full" in log
    assert "ROC-AUC (%)" in log and "AP (%)" in log
    config = json.loads((scripts / "ucf_v0.config.json").read_text(encoding="utf-8"))
    assert config["runner_sha256"] == hashlib.sha256(runner.encode()).hexdigest()
    assert config["config"]["OUTPUT"] == str(tmp_path / "output")


def test_table_keeps_every_column_aligned():
    table = REPORTING.render_table(
        ["Variant", "AUC", "Latency"],
        [["v0", "88.01993", "38.015"], ["normal_only", "9.1", "1000.12345"],
         ["v20", None, "0.00001"]], numeric=(1, 2),
    )
    lines = table.splitlines()
    assert len({len(line) for line in lines}) == 1
    columns = [[i for i, char in enumerate(line) if char == "|"]
               for line in lines if line.startswith("|")]
    assert all(positions == columns[0] for positions in columns)
    assert "N/A" in table
    assert "| 9.1 " not in table  # numbers are aligned to the right
    with pytest.raises(ValueError, match="number of columns"):
        REPORTING.render_table(["one"], [["one", "two"]])


@pytest.mark.parametrize("version", ["v0", "v7", "v20"])
def test_complete_notebook_startup_reaches_inference_boundary(tmp_path, version):
    """Run real setup/imports, stopping before model/data/network/GPU workloads."""
    nb = json.loads((NOTEBOOKS / f"V{int(version.removeprefix('v')):02d}.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    exec("".join(nb["cells"][1]["source"]), namespace)
    for index in ((2,) if version == "v20" else (3, 5, 6)):
        exec("".join(nb["cells"][index]["source"]), namespace)
    tree = ast.parse("".join(nb["cells"][3 if version == "v20" else 8]["source"]))
    # Keep all setup and functions, omitting only the final workload invocation.
    assert isinstance(tree.body[-1], ast.Assign)
    assert tree.body[-1].targets[0].id == "RESULT_ZIP"
    tree.body.pop()
    exec(compile(tree, "notebook-orchestrator", "exec"), namespace)
    namespace.update(INPUT=tmp_path, WORK=tmp_path, DATASETS=("ucf",))

    class InferenceBoundaryReached(Exception):
        pass

    reached = []

    def stop_before_inference(name, runner, config, scripts):
        reached.append(name)
        assert (scripts / "benchmark_reporting.py").is_file()
        assert (scripts / "latency_inference.py").is_file()
        raise InferenceBoundaryReached

    namespace["run_dataset"] = stop_before_inference
    if version == "v20":
        namespace["resolve_checkpoint"] = lambda *args: tmp_path / "model_ucf.pth"
    with pytest.raises(InferenceBoundaryReached):
        namespace["run_both"]()
    assert reached == ["ucf_v0"]
