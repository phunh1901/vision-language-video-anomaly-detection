import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from analysis import paired_bootstrap, ranked_counts, weighted_metrics
from anchor import DEFAULT_CONFIG, adapt_normal, anchor_forward, instrument_source
from sklearn.metrics import average_precision_score, roc_auc_score

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"



def tensors(dtype=torch.float32):
    # Two blocks from ONE video: final two positions are padding.
    v = torch.zeros(2, 8, 3, dtype=dtype)
    v[..., 0] = 1
    t = torch.zeros(2, 3, 2, dtype=dtype)
    t[:, :, 0] = torch.tensor([0.8, 0.6, 0], dtype=dtype)
    t[:, :, 1] = torch.tensor([0, 0, 1], dtype=dtype)
    c = torch.full((2, 8, 1), -5.0, dtype=dtype)
    a = v @ t / 0.07
    return v, t, c, a, [8, 6]


def test_zero_beta_is_bitwise_identity():
    *args, lengths = tensors()
    out, info, normal = adapt_normal(*args, lengths, {**DEFAULT_CONFIG, "beta_max": 0})
    assert torch.equal(out, args[-1]) and normal is None
    assert info["fallback"] == "beta_zero_control"


def test_insufficient_evidence_falls_back():
    *args, lengths = tensors()
    out, info, normal = adapt_normal(*args, lengths, DEFAULT_CONFIG)
    assert normal is None and torch.equal(out, args[-1])
    assert info["fallback"] == "insufficient_confident_snippets"


def test_padding_not_selected_and_anchor_is_unit():
    v, t, c, a, lengths = tensors()
    cfg = {**DEFAULT_CONFIG, "min_snippets": 2}
    _, info, normal = adapt_normal(v, t, c, a, lengths, cfg)
    assert 0 < info["beta"] <= cfg["beta_max"]
    assert max(info["selected_indices"]) < 14
    assert torch.allclose(normal.norm(dim=-1), torch.ones(2))
    v[1, 6:] = torch.tensor([-1, 0, 0.0])
    c[1, 6:] = 30
    a[1, 6:] = 30
    _, modified_info, modified_normal = adapt_normal(v, t, c, a, lengths, cfg)
    assert info == modified_info
    assert torch.equal(normal, modified_normal)


def test_high_score_video_never_forced_to_have_normal():
    v, t, c, a, lengths = tensors()
    c.fill_(5)
    _, info, normal = adapt_normal(v, t, c, a, lengths, DEFAULT_CONFIG)
    assert info["selected_count"] == 0 and normal is None


def test_no_gradient_through_anchor():
    v, t, c, a, lengths = tensors()
    v.requires_grad_()
    t.requires_grad_()
    c.requires_grad_()
    a.requires_grad_()
    out, _, normal = adapt_normal(v, t, c, a, lengths, {**DEFAULT_CONFIG, "min_snippets": 2})
    assert not out.requires_grad and not normal.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward_preserves_c_anomaly_logits_and_padding(dtype):
    v, t, c, a, lengths = tensors(dtype)

    class Model:
        training = False

        def forward(self, *_):
            self._v16_capture = (v, t, 0.07)
            return t[0].T, c, a

    model = Model()
    records = []
    _, out_c, out_a = anchor_forward(model, model.forward, v, None, ["Normal", "Fighting"],
                                    lengths, {**DEFAULT_CONFIG, "min_snippets": 2}, records)
    assert torch.equal(out_c, c)
    assert torch.equal(out_a[..., 1:], a[..., 1:])
    assert torch.equal(out_a[1, 6:], a[1, 6:])
    assert not torch.equal(out_a[0, :, 0], a[0, :, 0])
    assert len(records) == 1 and not hasattr(model, "_v16_capture")


def test_coherence_rejects_conflicting_background():
    v, t, c, a, lengths = tensors()
    v[0, 0] = -v[0, 0]
    v[0, 1] = -v[0, 1]
    _, info, normal = adapt_normal(v, t, c, a, lengths,
                                   {**DEFAULT_CONFIG, "min_snippets": 2, "bottom_fraction": 0.25})
    assert normal is None and info["fallback"] == "degenerate_anchor"


def test_stability_checks_real_neighbors_across_blocks():
    v, t, c, a, lengths = tensors()
    c[1, 0] = 5
    _, info, _ = adapt_normal(v, t, c, a, lengths,
                              {**DEFAULT_CONFIG, "bottom_fraction": 1, "min_snippets": 2})
    assert not ({7, 8, 9} & set(info["selected_indices"]))


SOURCE = '''
class CLIPVAD:
    def forward(self, visual, padding_mask, text, lengths):
        logits1 = visual.sum(-1, keepdim=True)
        visual_features_norm = visual / visual.norm(dim=-1, keepdim=True)
        text_features_norm = text / text.norm(dim=1, keepdim=True)
        logits2 = visual_features_norm @ text_features_norm.type(visual.dtype) / 0.07
        return text, logits1, logits2
'''


def test_instrumentation_retains_native_outputs():
    native, changed = {}, {}
    exec(SOURCE, native)
    exec(instrument_source(SOURCE), changed)
    v, t, _, _, lengths = tensors()
    model = changed["CLIPVAD"]()
    expected = native["CLIPVAD"]().forward(v, None, t, lengths)
    result = model.forward(v, None, t, lengths)
    assert all(torch.equal(a, b) for a, b in zip(expected, result, strict=True))
    x, y, temperature = model._v16_capture
    assert torch.equal(x @ y / temperature, result[2])


def test_instrumentation_rejects_drift():
    with pytest.raises(ValueError, match="alignment computation"):
        instrument_source(SOURCE.replace('        return text', '        logits2 = logits2 + 1\n        return text'))
    with pytest.raises(ValueError, match="temperature"):
        instrument_source(SOURCE.replace('/ 0.07', '* 0.07'))


def test_invalid_inputs_fail():
    v, t, c, a, lengths = tensors()
    v[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        adapt_normal(v, t, c, a, lengths, DEFAULT_CONFIG)


def test_weighted_metrics_match_explicit_frames_with_ties():
    scores = np.array([0.7, 0.2, 0.7, 0.4])
    pos = np.array([3, 0, 16, 1], dtype=float)
    weights = np.array([2, 0, 1, 3])
    truth, prediction = [], []
    for s, p, w in zip(scores, pos, weights, strict=True):
        truth.extend(([1] * int(p) + [0] * (16 - int(p))) * w)
        prediction.extend([s] * (16 * w))
    auc, ap = weighted_metrics(ranked_counts(scores, pos, 16 - pos), weights)
    assert auc == pytest.approx(roc_auc_score(truth, prediction))
    assert ap == pytest.approx(average_precision_score(truth, prediction))


def test_identical_bootstrap_zero():
    gt = np.tile(np.r_[np.ones(8), np.zeros(8)], 8)
    scores = np.linspace(0.1, 0.9, 8)
    out = paired_bootstrap(scores, scores, gt, np.array([0, 4, 8]), 20)
    assert out["valid"] == 20
    assert out["auc_ci95_pp"] == [0, 0] and out["ap_ci95_pp"] == [0, 0]


def test_notebook_and_embedded_sources_compile():
    path = NOTEBOOKS / "V16.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    scope = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"cell_{i}", "exec")
            if source.startswith(("ANCHOR_SOURCE", "ANALYSIS_SOURCE", "UCF_RUNNER", "XD_RUNNER")):
                exec(source, scope)
    for key, source in scope.items():
        if key != "__builtins__":
            compile(source, key, "exec")


def test_normalize_fixture():
    # Fixture features represent normalized embeddings, not raw image features.
    v, _, _, _, _ = tensors()
    assert torch.equal(v, F.normalize(v, dim=-1))


FAKE_RUNNER = '''
import json
import numpy as np
prefix = OUTPUT.parent.name
count = 290 if prefix == "ucf" else 800
base = OUTPUT / "full"
if prefix == "xd":
    base = base / "run_fixture"
base.mkdir(parents=True)
scores = np.tile([[0.1, 0.2], [0.9, 0.8]], (count, 1))
if VARIANT == "v16":
    scores[:, 1] *= 0.9
ids = np.array([str(i) for i in range(count)])
columns = ["classification", "alignment"] if prefix == "ucf" else ["branch_1", "branch_2"]
np.savez_compressed(base / "predictions.npz", scores=scores,
    snippet_offsets=np.arange(count+1)*2, frame_gt=np.tile(np.repeat([0,1],16),count),
    score_columns=np.array(columns), video_label=np.full(count,"anomaly"),
    **{("video_id" if prefix == "ucf" else "official_path"): ids})
metrics = {k: {"roc_auc":1.0,"ap":1.0,"frame_ap":1.0,"ano_auc":1.0} for k in columns}
report = {"run_kind":"full_checkpoint_inference", "video_count":count, "metrics":metrics,
    "frames_per_feature":16, "visual_block_length":256,
    **{k:"synthetic_fixture" for k in ["split_sha256","ground_truth_sha256",
    "evaluated_checkpoint_sha256","original_model_sha256"]}}
(base / "run.json").write_text(json.dumps(report))
(base / "anchor_diagnostics.json").write_text(json.dumps(
    [{"video_id":str(i),"beta":0.05} for i in range(count)]))
'''


def test_driver_full_two_datasets_and_partial_failure(tmp_path):
    from zipfile import ZipFile

    folder = (REPOSITORY / 'versions/_dev/v16')
    driver = (folder / "driver.py").read_text(encoding="utf-8")
    driver = driver.rsplit("RESULT_ZIP = run_both()", 1)[0]
    scope = {"Path": Path, "RUN_MODE": "full", "DATASETS": ("ucf", "xd"),
             "INPUT": tmp_path, "WORK": tmp_path, "ANCHOR_CONFIG": DEFAULT_CONFIG,
             "BOOTSTRAP_REPLICATES": 3, "SEED": 42,
             "ANCHOR_SOURCE": (folder / "anchor.py").read_text(encoding="utf-8"),
             "ANALYSIS_SOURCE": (folder / "analysis.py").read_text(encoding="utf-8"),
             "UCF_RUNNER": FAKE_RUNNER, "XD_RUNNER": FAKE_RUNNER,
             **{k: None for k in ["CLIP_WEIGHTS", "UCF_DATA_ROOT", "UCF_CHECKPOINT",
                                 "XD_DATA_ROOT", "XD_CHECKPOINT", "XD_FEATURE_ZIP"]}}
    exec(driver, scope)
    scope["V0_REFERENCE"] = {"ucf": ("classification", "roc_auc", 1.0),
                             "xd": ("branch_2", "frame_ap", 1.0)}
    archive = Path(scope["run_both"]())
    with ZipFile(archive) as z:
        summary = json.loads(z.read("summary.json"))
        assert all(r["status"] == "ok" for r in summary["runs"].values())
        assert len(summary["metrics"]) == 10
        assert z.testzip() is None
    scope["XD_RUNNER"] = 'raise ValueError("deliberate fixture failure")'
    with pytest.raises(RuntimeError, match="Some runs failed"):
        scope["run_both"]()
    outputs = sorted((tmp_path / "vadclip_v16").glob("run_*/summary.json"))
    summaries = [json.loads(p.read_text()) for p in outputs]
    assert any(s["runs"]["xd"]["status"] == "failed"
               and s["runs"]["ucf"]["status"] == "ok" for s in summaries)
