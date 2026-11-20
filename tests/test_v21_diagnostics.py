"""Flow contracts: temporal identity, bad stage detection, metric/error replay."""

import ast
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from fixtures.saved_author_forward import SavedAuthorForward
from sklearn.metrics import average_precision_score, roc_auc_score
from test_v21_native import NativeFixture, notebook_scope
from train_v21 import video_input
from v21_diagnostics import EvaluationTrace, check_batch
from v21_error_analysis import analyze_benchmark, probability_replay
from v21_native import NativePipeline


def test_audited_sampler_preserves_exact_output_and_rng():
    array = np.random.default_rng(1).normal(size=(39, 512)).astype("float32")
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    trace = {}
    left = video_input(array, 8, "cpu", a)
    right = video_input(array, 8, "cpu", b, trace)
    assert a.bit_generator.state == b.bit_generator.state
    for x, y in zip(left[:2], right[:2], strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    np.testing.assert_array_equal(right[0][0].numpy(), array[trace["selected_indices"]])
    assert trace["source_steps"] == 39 and len(trace["selected_indices"]) == 8


def trace_fixture():
    torch.manual_seed(7)
    model = NativeFixture().eval()
    x = torch.randn(2, 6, 512)
    x[1, 3:] = 0
    lengths = torch.tensor([6, 3])
    mask = torch.arange(6)[None] >= lengths[:, None]
    prompts = ["normal", "event", "other"]
    def dispatch():
        return model(x, mask, prompts, lengths)
    return model, x, lengths, mask, prompts, dispatch


def test_evaluation_trace_preserves_output_and_records_raw_logits_outside_timing(tmp_path):
    model, x, lengths, mask, prompts, dispatch = trace_fixture()
    with torch.no_grad():
        measured = dispatch()
    trace = EvaluationTrace(tmp_path, "ucf", "fixture")
    trace.observe("one-video", model, dispatch, x, lengths, mask, measured, 9, prompts=prompts)
    trace.finish()
    row = json.loads((tmp_path / "evaluation_flow.jsonl").read_text())
    assert row["replay_matches_timed_output"]
    assert {r["stage"] for r in row["stages"]} >= {"temporal", "linear", "classifier", "mlp1", "mlp2"}
    with np.load(tmp_path / "diagnostic_tensors.npz") as data:
        np.testing.assert_allclose(data["classification_logits"], measured[1].reshape(-1)[:9].numpy())
        assert data["alignment_logits"].shape == (9, 3)
        np.testing.assert_array_equal(data["block_index"], [0]*6+[1]*3)
    assert not model.linear._forward_hooks


def test_replay_detects_corruption_at_internal_stage_and_removes_hooks(tmp_path):
    model, x, lengths, mask, prompts, dispatch = trace_fixture()
    measured = dispatch()
    handle = model.linear.register_forward_hook(lambda m, args, value: value * float("nan"))
    trace = EvaluationTrace(tmp_path, "ucf", "fixture")
    with pytest.raises(ValueError, match="nonfinite"):
        trace.observe("bad-video", model, dispatch, x, lengths, mask, measured, 9, prompts=prompts)
    handle.remove()
    row = json.loads((tmp_path / "evaluation_flow.jsonl").read_text())
    assert row["video_id"] == "bad-video" and row["stage"] == "linear"
    assert row["status"] == "failed" and not model.linear._forward_hooks


@pytest.mark.parametrize("bad", ["length", "padding", "prompt", "measured"])
def test_evaluation_rejects_wrong_flow_contracts(tmp_path, bad):
    model, x, lengths, mask, prompts, dispatch = trace_fixture()
    measured = dispatch()
    if bad == "length":
        lengths = torch.tensor([5, 4])
    elif bad == "padding":
        x[1, 5, 0] = 1
    elif bad == "prompt":
        prompts[0] = "anomaly"
    else:
        measured = tuple(t+1 for t in measured)
    trace = EvaluationTrace(tmp_path, "ucf", "fixture")
    with pytest.raises(ValueError):
        trace.observe("bad", model, dispatch, x, lengths, mask, measured, 9, prompts=prompts)


@pytest.mark.parametrize("native_lengths", [None, [6, 6], [12], [[6, 3]]])
def test_xd_native_context_metadata_is_separate_from_verified_feature_export(tmp_path, native_lengths):
    model, x, _, _, prompts, _ = trace_fixture()
    x[1, 3:] = 2.5  # Legitimate context is observed, never rewritten by auditing.
    original = x.reshape(-1, 512)[:9].numpy().copy()
    sizes = None if native_lengths is None else torch.tensor(native_lengths)
    seen = []

    def dispatch():
        seen.append(sizes)
        return model(x, None, prompts, sizes)

    with torch.no_grad():
        measured = dispatch()
    trace = EvaluationTrace(tmp_path, "xd", "v0")
    trace.observe("native-xd", model, dispatch, x, sizes, None, measured, 9,
                  prompts=prompts, expected_features=original)
    trace.finish()
    row = json.loads((tmp_path / "evaluation_flow.jsonl").read_text())
    assert row["valid_lengths"] == [6, 3] and row["export_steps"] == 9
    assert row["model_lengths"] == native_lengths and row["input_matches_original_features"]
    assert not row["native_arguments_modified"] and not row["model_lengths_match_export"]
    assert row["nonzero_context_values_after_export"] == 3*512
    assert len(seen) == 2 and all(value is sizes for value in seen)
    with np.load(tmp_path / "diagnostic_tensors.npz") as data:
        np.testing.assert_array_equal(data["classification_logits"], measured[1].reshape(-1)[:9].numpy())


@pytest.mark.parametrize("bad", ["missing_features", "wrong_order", "wrong_shape", "capacity", "nan_lengths"])
def test_xd_distinct_native_lengths_do_not_disable_real_data_contracts(tmp_path, bad):
    model, x, _, _, prompts, _ = trace_fixture()
    sizes = torch.tensor([6., 6.])
    original = x.reshape(-1, 512)[:9].numpy().copy()

    def dispatch():
        return model(x, None, prompts, sizes)

    measured = dispatch()
    steps = 9
    if bad == "missing_features":
        original = None
    elif bad == "wrong_order":
        original[[0, 1]] = original[[1, 0]]
    elif bad == "wrong_shape":
        original = original[:8]
    elif bad == "capacity":
        steps = 13
    else:
        sizes[0] = float("nan")
    trace = EvaluationTrace(tmp_path, "xd", "v0")
    with pytest.raises(ValueError):
        trace.observe("bad-xd", model, dispatch, x, sizes, None, measured, steps,
                      prompts=prompts, expected_features=original)
    row = json.loads((tmp_path / "evaluation_flow.jsonl").read_text())
    assert row["status"] == "failed" and row["original_steps"] == steps
    if bad == "nan_lengths":
        assert "Nonfinite" in row["error"] and "nan" in row["model_lengths_repr"]


def test_every_batch_check_detects_label_sampling_mask_and_wrong_rounding():
    x = torch.zeros(2, 3, 512)
    lengths = torch.tensor([3, 2])
    valid = torch.arange(3)[None] < lengths[:, None]
    c, a = torch.full((2, 3, 1), 30.), torch.zeros(2, 3, 2)
    target = torch.tensor([[1., 0.], [0., 1.]])
    aux = {"adapter": x, "valid": valid}
    samples = [{"source_steps": 3, "selected_indices": [0, 1, 2]},
               {"source_steps": 9, "selected_indices": [0, 8]}]
    row = check_batch(x, lengths, target, c, a, aux, ["normal", "abnormal"], samples)
    assert row["videos"][0]["video_label_opposite_rounded_count"] == 3
    assert row["videos"][1]["video_label_opposite_rounded_count"] == 0
    samples[1]["selected_indices"] = [8, 0]
    with pytest.raises(ValueError, match="sample indices"):
        check_batch(x, lengths, target, c, a, aux, ["n", "a"], samples)
    samples[1]["selected_indices"] = [0, 8]
    aux["valid"] = ~valid
    with pytest.raises(ValueError, match="mask mismatch"):
        check_batch(x, lengths, target, c, a, aux, ["n", "a"], samples)
    aux["valid"] = valid
    target[0] = torch.ones(2)
    with pytest.raises(ValueError, match="targets"):
        check_batch(x, lengths, target, c, a, aux, ["n", "a"], samples)


def save_case(root, variant, c_logits, reverse_trace=False):
    root = Path(root) / variant
    root.mkdir(parents=True)
    z = torch.tensor(c_logits, dtype=torch.float32)
    align = torch.stack((torch.zeros_like(z), z), -1)
    scores = torch.stack((z.sigmoid(), 1-align.softmax(-1)[:, 0]), -1).numpy()
    gt = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.uint8)
    metrics = {}
    for col, name in enumerate(("classification", "alignment")):
        s = np.repeat(scores[:, col], 2)
        metrics[name] = {"roc_auc": float(roc_auc_score(gt, s)),
                         "ap": float(average_precision_score(gt, s))}
    np.savez_compressed(root / "predictions.npz", video_id=["normal", "abnormal"],
                        video_label=["Normal", "Abuse"], snippet_offsets=[0, 2, 4],
                        score_columns=["classification", "alignment"], scores=scores, frame_gt=gt)
    report = {"frames_per_feature": 2, "visual_block_length": 2, "metrics": metrics,
              "videos": [{"id": "normal", "steps": 2, "feature_sha256": "a"},
                         {"id": "abnormal", "steps": 2, "feature_sha256": "b"}]}
    path = root / "run.json"
    path.write_text(json.dumps(report))
    directory = root / "diagnostics"
    directory.mkdir()
    order = [2, 3, 0, 1] if reverse_trace else list(range(4))
    np.savez_compressed(directory / "diagnostic_tensors.npz",
                        video_id=["abnormal", "normal"] if reverse_trace else ["normal", "abnormal"],
                        snippet_offsets=[0, 2, 4], classification_logits=z.numpy()[order],
                        alignment_logits=align.numpy()[order], alignment_log_odds=z.double().numpy()[order],
                        scores=scores[order], adapter_norm=np.ones(4), feature_norm=np.ones(4), block_index=np.zeros(4))
    return path


def test_exact_error_intervals_raw_rank_loss_pairing_and_characteristics(tmp_path):
    v0 = save_case(tmp_path, "v0", [-3., -2., -1., 2.], reverse_trace=True)
    treatment = save_case(tmp_path, "control", [30., 30., 20., 30.])
    destination = tmp_path / "analysis"
    report = analyze_benchmark("ucf", {"v0": v0, "control": treatment}, destination)
    assert report["error_segment_count"] > 0
    assert report["trace_required"] and report["threshold_source"].startswith("fixed before test")
    import csv
    with (destination / "per_video_errors.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    abnormal = next(r for r in rows if r["variant"] == "control" and r["branch"] == "classification" and r["video_id"] == "abnormal")
    assert float(abnormal["within_video_auc"]) == .5
    assert float(abnormal["raw_within_video_auc"]) == 1
    assert abnormal["observed_stage"] == "probability_rounding"
    with (destination / "error_segments.csv").open() as stream:
        errors = list(csv.DictReader(stream))
    normal = next(r for r in errors if r["variant"] == "control" and r["video_id"] == "normal")
    assert normal["error"] == "FP" and normal["start_frame_0based"] == "0" and normal["end_frame_exclusive"] == "4"


def test_perfect_threshold_predictions_still_write_empty_error_tables(tmp_path):
    v0 = save_case(tmp_path, "v0", [-3., -2., -1., 2.])
    treatment = save_case(tmp_path, "control", [-3., -2., -1., 2.])
    destination = tmp_path / "analysis"
    summary = analyze_benchmark("ucf", {"v0": v0, "control": treatment}, destination)
    assert summary["error_segment_count"] == summary["stage_evidence_count"] == 0
    assert (destination / "error_segments.csv").read_text().count("\n") == 1


def test_fp32_softmax_reduction_witness_fails_old_check_but_passes_precision_budget(tmp_path):
    rng = np.random.default_rng(42)
    align = rng.normal(size=(4096, 14)).astype(np.float32)
    align[:, 0] += 8
    exponents = np.exp(align-align.max(-1, keepdims=True))
    denominator = np.zeros(len(align), dtype=np.float32)
    for col in exponents.T:
        denominator += col  # A different valid FP32 summation order.
    recorded = 1-exponents[:, 0]/denominator
    old_replay = (1-torch.from_numpy(align).softmax(-1)[:, 0]).numpy()
    assert not np.allclose(recorded, old_replay, rtol=1e-6, atol=1e-7)
    trace = {"classification_logits": np.zeros(len(align), dtype=np.float32),
             "alignment_logits": align, "scores": np.column_stack((np.full(len(align), .5), recorded)),
             "video_id": np.array(["fp32-witness"]), "snippet_offsets": np.array([0, len(align)])}
    original = trace["scores"].copy()
    receipt = probability_replay(trace, tmp_path)
    assert receipt["status"] == "passed" and not receipt["benchmark_scores_modified"]
    np.testing.assert_array_equal(trace["scores"], original)
    trace["scores"][7, 1] += .001
    with pytest.raises(ValueError, match="fp32-witness/alignment/snippet=7"):
        probability_replay(trace, tmp_path)
    failed = json.loads((tmp_path / "probability_replay.json").read_text())
    assert failed["status"] == "failed" and failed["examples"][0]["snippet_index"] == 7


@pytest.mark.parametrize("bad", ["feature_hash", "offsets", "score_export", "raw_logits", "raw_alignment_odds", "metric", "gt"])
def test_error_analysis_rejects_misattributed_or_corrupt_benchmark(tmp_path, bad):
    v0 = save_case(tmp_path, "v0", [-3., -2., -1., 2.])
    treatment = save_case(tmp_path, "control", [-3., -2., -1., 2.])
    if bad in {"feature_hash", "metric"}:
        report = json.loads(treatment.read_text())
        if bad == "feature_hash":
            report["videos"][0]["feature_sha256"] = "different"
        else:
            report["metrics"]["classification"]["roc_auc"] = .1
        treatment.write_text(json.dumps(report))
    else:
        file = treatment.parent / ("diagnostics/diagnostic_tensors.npz" if bad.startswith("raw_") else "predictions.npz")
        with np.load(file) as data:
            values = dict(data)
        if bad == "offsets":
            values["snippet_offsets"] = np.array([0, 3, 2])
        elif bad == "gt":
            values["frame_gt"][0] = 1
        elif bad == "raw_logits":
            values["classification_logits"] = np.zeros(4, dtype=np.float32)
        elif bad == "raw_alignment_odds":
            values["alignment_log_odds"] += 5
        else:
            values["scores"][0, 0] = .8
        np.savez_compressed(file, **values)
    with pytest.raises(ValueError):
        analyze_benchmark("ucf", {"v0": v0, "control": treatment}, tmp_path / "analysis")


class CapturedForwardFixture(SavedAuthorForward, NativeFixture):
    """Actual saved native forward; only encoder modules are CPU fixtures."""


class FixtureLatency:
    def __init__(self):
        self.rows = []
        self.calls = 0

    def measure(self, dispatch, *args):
        self.calls += 1
        self.rows.extend([{}, {}])
        return dispatch()


def test_built_ucf_runner_keeps_diagnostic_outside_timer_with_patched_native_forward(tmp_path):
    source = notebook_scope()["UCF_RUNNER"]
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"predict_video", "run_video"}]
    base = CapturedForwardFixture().eval()
    pipeline = NativePipeline(base).eval()
    base.forward = pipeline.forward
    latency = FixtureLatency()
    feature = np.random.default_rng(5).normal(size=(9, 512)).astype("float32")
    trace = EvaluationTrace(tmp_path, "ucf", "cosine_control")
    scope = {"torch": torch, "np": np, "FEATURE_DIM": 512, "VARIANT": "cosine_control",
             "LATENCY_REPEATS": 2, "latency": latency, "flow_trace": trace, "full_model": pipeline,
             "model": base, "device": "cpu", "CLASSES": ["normal", "event", "other"],
             "args": SimpleNamespace(visual_length=6), "feature_array": lambda path: feature,
             "get_batch_mask": lambda lengths, width: torch.arange(width)[None] >= lengths[:, None]}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "built_ucf", "exec"), scope)
    scores = scope["run_video"]({"path": "feature", "id": "actual-ucf-id", "steps": 9})
    trace.finish()
    assert scores.shape == (9, 2) and latency.calls == 1
    assert trace.ids == ["actual-ucf-id"]
    assert all(row["video_id"] == "actual-ucf-id" for row in latency.rows)
    with np.load(tmp_path / "diagnostic_tensors.npz") as data:
        np.testing.assert_array_equal(data["scores"], scores)


@pytest.mark.parametrize("space", [None, "native_cosine"])
def test_built_xd_runner_preserves_native_context_without_forward_recursion(tmp_path, space):
    torch.manual_seed(7)
    source = notebook_scope()["XD_RUNNER"]
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "count_forward")
    # Make closure bindings available in the isolated integration harness.
    class ClosureGlobals(ast.NodeTransformer):
        def visit_Nonlocal(self, node):
            return ast.copy_location(ast.Global(names=node.names), node)

    func = ClosureGlobals().visit(func)
    base = CapturedForwardFixture().eval()
    original = CapturedForwardFixture.forward
    pipeline = NativePipeline(base, native_forward=original).eval()
    latency = FixtureLatency()
    trace = EvaluationTrace(tmp_path, "xd", "v0" if space is None else "cosine_control")
    loads = []

    def load(*args, **kwargs):
        assert kwargs["native_forward"] is original
        loads.append(True)
        return pipeline, {"fixture": True}

    feature_path = tmp_path / "original.npy"
    scope = {"forwards": 0, "full_model": None, "full_provenance": None,
             "V21_SPACE": space, "V21_MODEL_PATH": "checkpoint", "V21_MODEL_SHA256": "hash",
             "checkpoint_hash": "v0", "load_native_model": load, "original_forward": original,
             "parameter_counts": {}, "count_parameters": lambda *args: {}, "VARIANT": "cosine_control",
             "inventory": [{"official_path": "actual-xd-id", "steps": 9}], "latency": latency,
             "flow_trace": trace, "RUN_MODE": "full", "getattr": getattr,
             "np": np, "feature_paths": [feature_path]}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "built_xd", "exec"), scope)
    base.forward = MethodType(scope["count_forward"], base)
    x = torch.randn(2, 6, 512)
    x[1, 3:] = 0
    np.save(feature_path, x.reshape(-1, 512)[:9].numpy())
    lengths = torch.tensor([6, 6])  # Native context differs from the 9-token export.
    mask = None
    with torch.no_grad():
        result = base(x, mask, ["normal", "event", "other"], lengths)
    trace.finish()
    assert scope["forwards"] == 1 and loads == ([] if space is None else [True]) and latency.calls == 1
    assert trace.ids == ["actual-xd-id"] and result[1].shape == (2, 6, 1)
    row = json.loads((tmp_path / "evaluation_flow.jsonl").read_text())
    assert row["model_lengths"] == [6, 6] and row["valid_lengths"] == [6, 3]
    assert row["input_matches_original_features"] and not row["native_arguments_modified"]
