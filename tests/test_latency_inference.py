"""Local tests for the exact profiler source embedded in all three notebooks."""

import ast
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"


ROOT = (REPOSITORY / 'versions/_dev')
SPEC = importlib.util.spec_from_file_location("vadclip_latency", SHARED / "latency_inference.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_warmup_rng_first_output_and_weighted_aggregation(tmp_path):
    profiler = MODULE.InferenceLatency(2, 3)
    torch.manual_seed(42)
    expected = torch.rand(1)
    torch.manual_seed(42)
    calls = []

    def forward():
        value = torch.rand(1)
        calls.append(value)
        return value

    # Video A: 10/20/30 ms -> median 20. Video B: 40/50/60 -> median 50.
    times = [0, .01, 1, 1.02, 2, 2.03, 3, 3.04, 4, 4.05, 5, 5.06]
    with patch.object(MODULE.time, "perf_counter", side_effect=times):
        output = profiler.measure(forward, "cpu", "a", 10, (1, 10, 512), torch.float32)
        profiler.measure(forward, "cpu", "b", 20, (1, 20, 512), torch.float32)
    assert torch.equal(output, expected)
    assert len(calls) == 8  # 2 warmups once, 2 * 3 measured calls
    assert len(profiler.rows) == 6
    result = profiler.save(tmp_path, "ucf", "v0", "full", "cpu")
    assert result["video_count"] == 2
    assert result["measurement_count"] == 6
    assert result["snippets"] == 30
    assert result["mean_ms_per_video"] == pytest.approx(35)
    assert result["median_ms_per_video"] == pytest.approx(35)
    assert result["p95_ms_per_video"] == pytest.approx(48.5)
    assert result["ms_per_snippet"] == pytest.approx(70 / 30)
    assert result["snippets_per_second"] == pytest.approx(30 / .07)
    assert len((tmp_path / "latency_per_video.csv").read_text().splitlines()) == 7
    assert MODULE.collect_latency(tmp_path) == [result]
    assert (tmp_path / "latency_comparison.csv").is_file()


def test_cuda_synchronization_brackets_each_measurement():
    profiler = MODULE.InferenceLatency(0, 2)
    events = []

    def clock():
        events.append("clock")
        return len(events) / 100

    def forward():
        events.append("forward")
        return "first-logits"

    with (
        patch.object(torch.random, "fork_rng"),
        patch.object(torch.cuda, "synchronize", side_effect=lambda _: events.append("sync")),
        patch.object(MODULE.time, "perf_counter", side_effect=clock),
    ):
        result = profiler.measure(forward, "cuda:0", "a", 1, (1, 1, 512), torch.float32)
    assert result == "first-logits"
    assert events == ["sync"] + ["sync", "clock", "forward", "sync", "clock"] * 2


@pytest.mark.parametrize("warmup,repeats", [(-1, 3), (3, 0), (True, 3), (3, 1.5)])
def test_invalid_config(warmup, repeats):
    with pytest.raises(ValueError):
        MODULE.InferenceLatency(warmup, repeats)


def test_notebooks_embed_identical_helper_and_compile():
    helper = (SHARED / "latency_inference.py").read_text(encoding="utf-8")
    for version in ("v0", "v7", "v20"):
        nb = json.loads((NOTEBOOKS / f"V{int(version.removeprefix('v')):02d}.ipynb").read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{version}:cell{i}", "exec")
        namespace = {}
        for index in ((2,) if version == "v20" else (3, 5, 6)):
            exec("".join(nb["cells"][index]["source"]), namespace)
        embedded = (namespace["SOURCES"]["latency_inference.py"] if version == "v20"
                    else namespace["LATENCY_SOURCE"])
        assert embedded == helper
        for name in ("UCF_RUNNER", "XD_RUNNER"):
            code = namespace[name]
            compile(code, f"{version}:{name}", "exec")
            assert "latency.measure(" in code
            assert "latency.save(" in code
            assert '"latency": latency_summary' in code
            assert "def dispatch():" in code
        if version == "v20":
            compile(namespace["TRAINER_SOURCE"], "trainer", "exec")
            for name, code in namespace["SOURCES"].items():
                compile(code, name, "exec")
        else:
            compile(namespace["WINDOW_SOURCE"], "v7_windows.py", "exec")
        if version == "v0":
            assert 'for variant in ("v0",):' in "".join(nb["cells"][8]["source"])


@pytest.mark.parametrize("version,variant", [("v0", "v0"), ("v7", "v7"), ("v20", "v20")])
def test_embedded_ucf_prediction_preserves_scores_and_valid_steps(version, variant):
    nb = json.loads((NOTEBOOKS / f"V{int(version.removeprefix('v')):02d}.ipynb").read_text(encoding="utf-8"))
    payload = {}
    exec("".join(nb["cells"][2 if version == "v20" else 5]["source"]), payload)
    tree = ast.parse(payload["UCF_RUNNER"])
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "predict_video")
    profiler = MODULE.InferenceLatency(1, 2)
    calls = []

    def model(visual, mask, text, lengths):
        calls.append(visual.shape)
        c = visual[..., :1] * 0 + 0.5
        a = torch.cat([c * 0, c], dim=-1)
        return c, c, a

    def overlap(forward, *args):
        return forward(*args[:-1])

    namespace = {
        "np": np, "torch": torch, "FEATURE_DIM": 512, "CLASSES": ["Normal", "Other"],
        "VARIANT": variant, "LATENCY_REPEATS": 2, "latency": profiler,
        "overlap_forward": overlap,
        "get_batch_mask": lambda lengths, width: torch.zeros((len(lengths), width)),
        "bundle": None, "apply_head": lambda bundle, v, lengths, c, a, branch: (c, a),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "embedded", "exec"), namespace)
    for steps in (3, 11):
        result = namespace["predict_video"](
            np.zeros((steps, 512), dtype=np.float32), model, torch.device("cpu"), 8,
        )
        expected = np.full((steps, 2), torch.tensor(.5).sigmoid().item())
        assert result.shape == (steps, 2)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert calls == [(1, 8, 512)] * 3 + [(2, 8, 512)] * 2
    assert [row["snippets"] for row in profiler.rows] == [3, 3, 11, 11]


@pytest.mark.parametrize("version", ["v0", "v7", "v20"])
@pytest.mark.parametrize("smoke_only", [True, False])
def test_ucf_latency_export_initializes_directory_first(tmp_path, version, smoke_only):
    """Execute the embedded export statements in their real top-level order."""
    nb = json.loads((NOTEBOOKS / f"V{int(version.removeprefix('v')):02d}.ipynb").read_text(encoding="utf-8"))
    payload = {}
    exec("".join(nb["cells"][2 if version == "v20" else 5]["source"]), payload)
    tree = ast.parse(payload["UCF_RUNNER"])
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"output_dir", "latency_summary"}
            for target in node.targets
        ):
            nodes.append(node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and node.value.func.value.id == "output_dir" and node.value.func.attr == "mkdir"):
            nodes.append(node)
    assert len(nodes) == 3
    profiler = MODULE.InferenceLatency(0, 1)
    profiler.rows = [{"video_id": "example", "repeat": 0, "snippets": 10,
                      "input_shape": "(1, 10, 512)", "input_dtype": "torch.float32",
                      "latency_ms": 20., "ms_per_snippet": 2.}]
    namespace = {
        "OUTPUT": tmp_path, "SMOKE_ONLY": smoke_only, "VARIANT": version,
        "device": torch.device("cpu"), "latency": profiler,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "embedded-export", "exec"), namespace)
    directory = tmp_path / ("smoke" if smoke_only else "full")
    summary = json.loads((directory / "latency_summary.json").read_text(encoding="utf-8"))
    assert summary == namespace["latency_summary"]
    assert summary["video_count"] == 1
    assert summary["ms_per_snippet"] == 2.
    assert (directory / "latency_per_video.csv").is_file()


@pytest.mark.parametrize("version", ["v0", "v7", "v20"])
def test_download_cell_copies_archive_and_links_relative_name(tmp_path, monkeypatch, version):
    import sys
    import zipfile
    from types import ModuleType, SimpleNamespace

    source = tmp_path / "results" / "run_example" / "latency_results.zip"
    source.parent.mkdir(parents=True)
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("latency_comparison.csv", "dataset,variant\nucf,v0\n")
    notebook_directory = tmp_path / "notebook"
    notebook_directory.mkdir()
    monkeypatch.chdir(notebook_directory)
    displayed = []
    # Kaggle supplies IPython; local runtime has no notebook renderer.
    display_module = ModuleType("IPython.display")
    display_module.display = displayed.append
    display_module.FileLink = lambda path: SimpleNamespace(path=path)
    monkeypatch.setitem(sys.modules, "IPython", ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    nb = json.loads((NOTEBOOKS / f"V{int(version.removeprefix('v')):02d}.ipynb").read_text(encoding="utf-8"))
    namespace = {"RESULT_ZIP": source}
    exec("".join(nb["cells"][-1]["source"]), namespace)
    destination = namespace["download_zip"]
    assert destination.parent == notebook_directory
    assert destination.read_bytes() == source.read_bytes()
    assert displayed[0].path == destination.name
    assert not Path(displayed[0].path).is_absolute()
