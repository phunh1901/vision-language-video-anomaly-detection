"""Numerical, gradient, training and standalone integration tests; synthetic CPU fixtures."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import v21_data as data
from torch import nn
from torch.nn import functional as F
from train_v21 import fit, video_input
from v21_geometry import cone_energy, expmap0, logmap0, poincare_distance
from v21_model import (
    SCHEMA,
    FullPipeline,
    full_loss,
    load_full_model,
    pooled_binary_loss,
    topk_size,
)

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

HERE = (REPOSITORY / 'versions/_dev/v21')
GEOMETRY = {"dimension": 4, "curvature": .5, "scale": .7, "temperature": .07, "chunk_size": 2}
LOSS = {"psi": 1e-3, "phi": .1, "cone_aperture_scale": .1,
        "entail_regularizer": .01, "separation_margin": 1.}


class FakeBase(nn.Module):
    """Explicit test fixture of the expected API, not upstream author code."""
    def __init__(self, dimension=4):
        super().__init__()
        self.temporal = nn.Linear(dimension, dimension)
        self.text_prompt = nn.Embedding(3, dimension)
        self.clipmodel = nn.Linear(dimension, dimension)
        self.mlp1 = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU())
        self.classifier = nn.Linear(dimension, 1)
        self.mlp2 = nn.Linear(dimension, dimension)

    def encode_video(self, images, padding_mask, lengths):
        return self.temporal(images)

    def encode_textprompt(self, text):
        return self.clipmodel(self.text_prompt.weight[:len(text)])


def inputs():
    torch.manual_seed(91)
    return torch.randn(2, 6, 4), torch.tensor([6, 3]), torch.tensor([[1., 0., 0.], [0., 1., 0.]])


def mobius_reference(x, y, c):
    """Independent direct Eq.(3), not the asinh implementation used by V21."""
    u, v = -x, y
    dot = (u * v).sum(-1, keepdim=True)
    unorm, vnorm = u.square().sum(-1, keepdim=True), v.square().sum(-1, keepdim=True)
    numerator = (1 + 2 * c * dot + c * vnorm) * u + (1 - c * unorm) * v
    denominator = 1 + 2 * c * dot + c.square() * unorm * vnorm
    added = numerator / denominator
    return 2 / c.sqrt() * torch.atanh(c.sqrt() * torch.linalg.vector_norm(added, dim=-1))


@pytest.mark.parametrize("curvature", [.1001, .5, .9999])
def test_maps_distance_and_independent_mobius(curvature):
    torch.manual_seed(5)
    c = torch.tensor(curvature, dtype=torch.double)
    v, w = torch.randn(4, 7, dtype=torch.double) * .12, torch.randn(4, 7, dtype=torch.double) * .12
    x, y = expmap0(v, c), expmap0(w, c)
    torch.testing.assert_close(logmap0(x, c), v, atol=1e-12, rtol=1e-12)
    actual = poincare_distance(x, y, c)
    torch.testing.assert_close(actual, mobius_reference(x, y, c), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual, poincare_distance(y, x, c))
    torch.testing.assert_close(poincare_distance(x, x, c), torch.zeros(4, dtype=torch.double))
    torch.testing.assert_close(poincare_distance(torch.zeros_like(x), x, c),
                               2 * torch.linalg.vector_norm(v, dim=-1))


def test_geometry_gradcheck_and_origin_boundary_are_finite():
    c = torch.tensor(.5, dtype=torch.double, requires_grad=True)
    v = (torch.randn(2, 3, dtype=torch.double) * .1).requires_grad_()
    w = (torch.randn(2, 3, dtype=torch.double) * .1).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b, k: poincare_distance(
        expmap0(a, k), expmap0(b, k), k), (v, w, c), atol=1e-5)
    zero = torch.zeros(2, 3, requires_grad=True)
    loss = poincare_distance(expmap0(zero, c.float()), expmap0(zero, c.float()), c.float()).sum()
    loss.backward()
    assert torch.isfinite(zero.grad).all()
    huge = torch.tensor([[1e4, -1e4, 1e4]], requires_grad=True)
    x = expmap0(huge, c.float())
    value = poincare_distance(x, torch.zeros_like(x), c.float()).sum()
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(huge.grad).all()
    assert (c.float() * x.square().sum(-1) < 1).all()


def test_radial_information_is_not_cosine_and_sphere_order_is_monotone():
    c = torch.tensor(.5, dtype=torch.double)
    q = torch.tensor([[.2, 0.]], dtype=torch.double)
    near = torch.tensor([[.25, 0.]], dtype=torch.double)
    far = torch.tensor([[.8, 0.]], dtype=torch.double)
    assert F.cosine_similarity(q, near) == F.cosine_similarity(q, far)
    assert poincare_distance(q, near, c) < poincare_distance(q, far, c)
    angles = torch.linspace(.1, 3., 20, dtype=torch.double)
    points = torch.stack((angles.cos(), angles.sin()), dim=-1) * .4
    origin_direction = torch.tensor([[.4, 0]], dtype=torch.double)
    similarity = F.cosine_similarity(origin_direction, points)
    hyper_similarity = -poincare_distance(origin_direction, points, c)
    assert torch.equal(similarity.argsort(), hyper_similarity.argsort())


def test_group_split_excludes_source_overlap_and_unknown_labels():
    rows = [{"path": f"source{i}__#clip__0.npy", "label": "A" if i % 2 else "B"}
            for i in range(40)]
    test = [{"path": "source0__#different__0.npy", "label": "B"}]
    features = {data.video_id(r["path"]): {} for r in rows}

    def encode(labels):
        return torch.tensor([[1., 0.] if x == "A" else [0., 1.] for x in labels])

    ids, groups, _, _, train, dev, excluded, overlap = data.dataset_split(
        rows, test, features, encode, "xd", 42)
    assert overlap == 1 and excluded.tolist() == [0]
    assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[dev])
    assert 0 not in train and 0 not in dev and len(ids) == 40
    with pytest.raises(ValueError, match="rỗng"):
        data.dataset_split(rows, test, features, lambda labels: torch.zeros(len(labels), 2), "xd", 42)


def test_full_model_updates_adapter_prompt_and_both_branches_but_freezes_clip():
    x, lengths, targets = inputs()
    model = FullPipeline(FakeBase(), "poincare", GEOMETRY)
    frozen = copy.deepcopy(model.base.clipmodel.state_dict())
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    model.train()
    assert not model.base.clipmodel.training
    _, c, a, aux = model.forward_with_aux(x, None, ["normal", "fight", "abuse"], lengths)
    original_c, original_a = c.detach().clone(), a.detach().clone()
    losses, parts = full_loss(c, a, targets, lengths, aux, "poincare", LOSS)
    torch.testing.assert_close(losses, parts["bce"] + parts["alignment"] +
                               LOSS["psi"] * parts["entailment"] + LOSS["phi"] * parts["contrastive"])
    losses.mean().backward()
    for module in (model.base.temporal, model.base.text_prompt, model.base.mlp1, model.geometry):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.base.clipmodel.parameters())
    optimizer.step()
    _, c, a = model(x, None, ["normal", "fight", "abuse"], lengths)
    assert not torch.allclose(c, original_c) and not torch.allclose(a, original_a)
    for name, value in frozen.items():
        assert torch.equal(value, model.base.clipmodel.state_dict()[name])
    assert not torch.equal(before["base.text_prompt.weight"], model.base.text_prompt.weight)
    assert not torch.equal(before["geometry.prototype"], model.geometry.prototype)


@pytest.mark.parametrize("target", [0., 1.])
def test_logdomain_bce_matches_probability_objective(target):
    z = torch.tensor([-3., .7, 1.2, -2.], requires_grad=True)
    y = torch.tensor(target)
    actual = pooled_binary_loss(z, y, 3)
    reference = F.binary_cross_entropy(z.sigmoid().topk(3).values.mean(), y)
    torch.testing.assert_close(actual, reference)
    a = torch.autograd.grad(actual, z, retain_graph=True)[0]
    b = torch.autograd.grad(reference, z)[0]
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("value,target", [(-20., 1.), (-1000., 1.), (1000., 0.)])
def test_logdomain_bce_extreme_logits_still_have_finite_nonzero_gradients(value, target):
    z = torch.full((17,), value, requires_grad=True)
    loss = pooled_binary_loss(z, torch.tensor(target), 17)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > .99


def test_nonfinite_logits_rejected_before_bce_cuda_kernel():
    with pytest.raises(ValueError, match="classification logits"):
        pooled_binary_loss(torch.tensor([float("nan")]), torch.tensor(1.), 1)


def test_padding_does_not_affect_prompt_or_video_loss():
    x, lengths, targets = inputs()
    model = FullPipeline(FakeBase(), "poincare", GEOMETRY).eval()
    _, c, a, aux = model.forward_with_aux(x, None, ["n", "a", "b"], lengths)
    loss, _ = full_loss(c, a, targets, lengths, aux, "poincare", LOSS)
    x[1, 3:] = 1234
    _, c2, a2, aux2 = model.forward_with_aux(x, None, ["n", "a", "b"], lengths)
    loss2, _ = full_loss(c2, a2, targets, lengths, aux2, "poincare", LOSS)
    torch.testing.assert_close(loss, loss2)
    torch.testing.assert_close(aux["text"], aux2["text"])
    assert topk_size(256, True) == 17 and topk_size(256, False) == 1
    with pytest.raises(ValueError, match="Unknown/conflicting"):
        full_loss(c, a, torch.zeros_like(targets), lengths, aux, "poincare", LOSS)


def test_accumulation_matches_physical_batch_without_batch_coupling():
    x, lengths, targets = inputs()
    torch.manual_seed(15)
    model = FullPipeline(FakeBase(), "poincare", GEOMETRY)
    other = copy.deepcopy(model)
    _, c, a, aux = model.forward_with_aux(x, None, ["n", "a", "b"], lengths)
    full_loss(c, a, targets, lengths, aux, "poincare", LOSS)[0].mean().backward()
    for i in range(2):
        _, c, a, aux = other.forward_with_aux(x[i:i+1], None, ["n", "a", "b"], lengths[i:i+1])
        (full_loss(c, a, targets[i:i+1], lengths[i:i+1], aux, "poincare", LOSS)[0].sum()/2).backward()
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=2e-4)


def test_cone_angle_matches_independent_closed_form():
    # Ganea exterior angle, curvature-one coordinates, independently of Mobius.
    parent = torch.tensor([[.4, .1]], dtype=torch.double, requires_grad=True)
    child = torch.tensor([[.7, .2]], dtype=torch.double, requires_grad=True)
    c = torch.tensor(.5, dtype=torch.double, requires_grad=True)
    p, x = parent*c.sqrt(), child*c.sqrt()
    p2, x2 = p.square().sum(-1), x.square().sum(-1)
    dot = (p*x).sum(-1)
    cosine = (dot*(1+p2)-p2*(1+x2)) / (
        p.norm(dim=-1)*(x-p).norm(dim=-1)*(1+p2*x2-2*dot).sqrt())
    angle = cosine.clamp(-1+1e-6, 1-1e-6).acos()
    aperture = (.1*(1-p2)/p.norm(dim=-1)).asin()
    expected = (angle-aperture).clamp_min(0)-.01*angle
    actual = cone_energy(child, parent, c)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    actual.sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in (parent, child, c))


def test_distance_chunking_and_checkpoint_schema_hash(tmp_path):
    x, lengths, _ = inputs()
    original_base = FakeBase()
    base_state = copy.deepcopy(original_base.state_dict())
    model = FullPipeline(original_base, "poincare", GEOMETRY).eval()
    with torch.no_grad():
        _, c, a = model(x, None, ["n", "a", "b"], lengths)
    path = tmp_path / "best.pth"
    torch.save({"schema": SCHEMA, "space": "poincare", "geometry_config": GEOMETRY,
                "initial_v0_sha256": "original", "epoch": 1,
                "downstream": model.downstream_state()}, path)
    fresh = FakeBase()
    fresh.load_state_dict(base_state)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, receipt = load_full_model(fresh, "poincare", GEOMETRY, path, digest, "original", "cpu")
    _, c2, a2 = loaded(x, None, ["n", "a", "b"], lengths)
    torch.testing.assert_close(c, c2)
    torch.testing.assert_close(a, a2)
    assert receipt["trained_checkpoint_sha256"] == digest
    loaded.geometry.chunk_size = 100
    torch.testing.assert_close(loaded(x, None, ["n", "a", "b"], lengths)[2], a2)
    with pytest.raises(ValueError, match="SHA-256"):
        load_full_model(FakeBase(), "poincare", GEOMETRY, path, "wrong", "original", "cpu")


def test_video_input_sampling_and_cpu_fit_saved_selection(tmp_path):
    torch.set_num_threads(2)
    array = np.arange(512*40, dtype=np.float32).reshape(40, 512) / 10000
    visual, lengths, count = video_input(array, 16, "cpu", np.random.default_rng(42))
    assert visual.shape == (1, 16, 512) and count == 16 and lengths.tolist() == [16]
    visual, lengths, count = video_input(array, 16, "cpu")
    assert visual.shape == (3, 16, 512) and count == 40 and lengths.tolist() == [16, 16, 8]

    class Pack:
        features = {str(i): i for i in range(8)}

        def array(self, index):
            return np.ones((5, 512), np.float32) * (.01 + index*.01)

    ids = [str(i) for i in range(8)]
    labels = torch.tensor([[1., 0., 0.] if i % 2 == 0 else [0., 1., 0.] for i in range(8)])
    abnormal = labels[:, 0].numpy() == 0
    config = {"epochs": 2, "learning_rate": 5e-4, "weight_decay": 5e-5,
              "gradient_clip": 1., "amp": False, "phi": {"ucf": .1, "xd": 1e-4},
              "loss": {k: v for k, v in LOSS.items() if k != "phi"}}
    result = fit(FakeBase(512), Pack(), ids, labels, abnormal, np.arange(4), np.arange(4, 8),
                 SimpleNamespace(visual_length=16), ["n", "a", "b"],
                 lambda lengths, width: torch.arange(width)[None] >= lengths[:, None],
                 False, config, {**GEOMETRY, "dimension": 512}, tmp_path/"v21", 42, 1,
                 torch.device("cpu"), "ucf", "original")
    assert len(result["history"]) == 2 and result["optimizer"] == "Adam"
    assert result["preflight"]["calibrated"] is True
    assert result["preflight"]["optimizer_updates"] == 0
    assert result["preflight"]["test_data_used"] is False
    assert (tmp_path/"v21"/"preflight.json").is_file()
    assert result["selection_branch"] == "classification"
    assert result["history"][-1]["next_learning_rate"] == 0
    assert result["history"][0]["optimizer_updates"] > 0
    assert all(result["first_gradient_groups"][k] for k in
               ("temporal_adapter", "prompt_fusion", "geometry_classifier"))
    checkpoint = tmp_path / result["checkpoint_file"]
    state = torch.load(checkpoint, weights_only=True)
    assert state["epoch"] == result["best_epoch"] and state["schema"] == SCHEMA
    assert not any(k.startswith("base.clipmodel.") for k in state["downstream"])


def test_notebook_compile_settings_and_startup_no_cosine_training(tmp_path):
    notebook = json.loads((HERE / "fixtures/recipe_notebooks/run_vadclip_v21_A.ipynb.json").read_text(encoding="utf-8"))
    namespace = {}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{i}", "exec")
    for i in (1, 2):
        exec("".join(notebook["cells"][i]["source"]), namespace)
    assert namespace["TRAIN_CONFIG"]["epochs"] == 16 and namespace["HALF_BATCH"]*2 == 64
    assert namespace["TRAIN_CONFIG"]["two_stage"]["adapter_lr"] == 2e-5
    for name, source in namespace["SOURCES"].items():
        compile(source, name, "exec")
        if (HERE/name).is_file():
            assert source == (HERE/name).read_text(encoding="utf-8")
    for key in ("UCF_RUNNER", "XD_RUNNER"):
        compile(namespace[key], key, "exec")
        assert "load_full_model" in namespace[key]
        assert "initial_v0_checkpoint_sha256" in namespace[key]
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    tree.body.pop()
    exec(compile(tree, "orchestrator", "exec"), namespace)
    namespace.update(INPUT=tmp_path, WORK=tmp_path, DATASETS=("ucf",))
    namespace["resolve_checkpoint"] = lambda *args: tmp_path/"model_ucf.pth"

    class BoundaryReached(Exception):
        pass

    def stop(name, runner, config, scripts):
        assert name == "ucf_v0" and config["V21_SPACE"] is None
        assert (scripts/"v21_model.py").is_file()
        raise BoundaryReached

    namespace["run_dataset"] = stop
    with pytest.raises(RuntimeError, match="Incomplete V21 run"):
        namespace["run_both"]()
    assert namespace["LAST_RESULT_ZIP"].is_file()
    assert 'evaluate("poincare_untrained"' not in namespace["SOURCES"]["orchestrator_v21.py"]
    assert 'for name in ("cosine_control"' not in namespace["SOURCES"]["orchestrator_v21.py"]


def orchestrator_namespace():
    notebook = json.loads((HERE / "fixtures/recipe_notebooks/run_vadclip_v21_A.ipynb.json").read_text(encoding="utf-8"))
    namespace = {}
    for i in (1, 2):
        exec("".join(notebook["cells"][i]["source"]), namespace)
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    tree.body.pop()
    exec(compile(tree, "orchestrator", "exec"), namespace)
    return namespace


def test_orchestrator_both_datasets_only_v0_train_poincare_and_saved_zip(tmp_path):
    from zipfile import ZipFile

    namespace = orchestrator_namespace()
    namespace.update(INPUT=tmp_path, WORK=tmp_path)
    jobs = []
    namespace["resolve_checkpoint"] = lambda *args: tmp_path/"original.pth"
    namespace["load_module"] = lambda *args: SimpleNamespace(
        collect_reports=lambda *a: None, collect_latency=lambda *a: None)

    def fake_job(name, source, config, scripts):
        # Scheduling/packaging fixture only; no model inference is claimed here.
        jobs.append(name)
        output = Path(config["OUTPUT"])
        output.mkdir(parents=True)
        if name.endswith("_train"):
            assert config["HALF_BATCH"] == 32 and config["TRAIN_CONFIG"]["epochs"] == 16
            path = output/"training.json"
            weights = output/"v21"/"best.pth"
            weights.parent.mkdir()
            weights.write_bytes(b"synthetic checkpoint fixture")
            (weights.parent/"last_training_state.pt").write_bytes(b"excluded fixture")
            path.write_text(json.dumps({"variants": {"v21": {
                "space": "poincare", "checkpoint_file": "v21/best.pth",
                "checkpoint_sha256": "synthetic"}}}), encoding="utf-8")
        else:
            if name.endswith("_v21"):
                assert config["V21_SPACE"] == "poincare" and config["V21_MODEL_SHA256"] == "synthetic"
                assert Path(config["V21_MODEL_PATH"]).is_file()
            else:
                assert config["V21_SPACE"] is None
            path = output/"run.json"
            path.write_text('{}', encoding="utf-8")
        return path

    namespace["run_dataset"] = fake_job
    archive = namespace["run_both"]()
    assert jobs == ["ucf_v0", "ucf_train", "ucf_v21", "xd_v0", "xd_train", "xd_v21"]
    with ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert sum(p.endswith("best.pth") for p in names) == 2
        assert not any(p.endswith("last_training_state.pt") for p in names)
        summary = json.loads(bundle.read("summary.json"))
        assert summary["cosine_control_trained"] is False
        assert all(item["status"] == "ok" for item in summary["runs"].values())


@pytest.mark.parametrize("prefix", ["ucf", "xd"])
def test_full_comparison_allows_changed_c_and_trained_hash_but_rejects_grid(tmp_path, prefix):
    ns = orchestrator_namespace()
    branches = ("classification", "alignment") if prefix == "ucf" else ("branch_1", "branch_2")
    ap_key = "ap" if prefix == "ucf" else "frame_ap"
    paths = {}
    for variant in ("v0", "v21"):
        folder = tmp_path/variant
        folder.mkdir()
        path = folder/"run.json"
        report = {"initial_v0_checkpoint_sha256": "original", "evaluated_checkpoint_sha256": variant,
                  "split_sha256": "split", "ground_truth_sha256": "gt",
                  "visual_block_length": 256, "frames_per_feature": 16,
                  "metrics": {branch: {"roc_auc": .8801992605166816,
                                       ap_key: .8450448809004643} for branch in branches}}
        path.write_text(json.dumps(report), encoding="utf-8")
        arrays = {"video_id" if prefix == "ucf" else "official_path": np.array(["a", "b"]),
                  "snippet_offsets": np.array([0, 2, 4]), "score_columns": np.array(["C", "A"]),
                  "frame_gt": np.repeat([0, 0, 1, 1], 16), "video_label": np.array([0, 1]),
                  "scores": np.full((4, 2), .1 if variant == "v0" else .5)}
        np.savez(folder/"predictions.npz", **arrays)
        paths[variant] = path
    calls = []

    def bootstrap(left, right, *args):
        calls.append((left.copy(), right.copy()))
        return {"auc_ci95_pp": [-1., 2.], "ap_ci95_pp": [-2., 3.]}

    rows = ns["verify_comparison"](prefix, paths, SimpleNamespace(paired_bootstrap=bootstrap))
    assert len(rows) == 4 and len(calls) == 2
    assert set(r["branch"] for r in rows) == set(branches)
    arrays["snippet_offsets"] = np.array([0, 1, 4])
    np.savez(paths["v21"].with_name("predictions.npz"), **arrays)
    with pytest.raises(ValueError, match="Mismatched evaluator grid"):
        ns["verify_comparison"](prefix, paths, SimpleNamespace(paired_bootstrap=bootstrap))
