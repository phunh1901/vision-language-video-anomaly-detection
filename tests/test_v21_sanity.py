"""Regression checks for diagnosed failures; no author/GPU benchmark claims."""

import copy
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from test_v21 import LOSS, FakeBase, inputs, orchestrator_namespace
from test_v21_paper import profile
from v21_geometry import cone_energy, mobius_add
from v21_health import joint_optimizer, learning_health, probe_indices
from v21_model import FullPipeline, full_loss


def test_near_boundary_cone_preserves_visual_gradient_with_active_hinge():
    c = torch.tensor(.5, dtype=torch.float64)
    parent = torch.tensor([.999935 / .5**.5, 0.], dtype=torch.float64, requires_grad=True)
    child = torch.tensor([.3, .1], dtype=torch.float64, requires_grad=True)
    energy, stats = cone_energy(child, parent, c, diagnostics=True)
    energy.backward()
    assert stats["legacy_acos_would_clip_fraction"] == 1
    assert stats["hinge_active_fraction"] == 1
    assert stats["asin_clip_fraction"] == 0
    assert child.grad.norm() > 0 and torch.isfinite(child.grad).all()
    assert torch.autograd.gradcheck(lambda x: cone_energy(x, parent.detach(), c),
                                    (child.detach().requires_grad_(),), eps=1e-5, atol=1e-6)


def test_atan2_energy_matches_unclipped_acos_on_interior():
    c = torch.tensor(.5, dtype=torch.float64)
    p = torch.tensor([[.6, .1], [.5, -.2]], dtype=torch.float64)
    v = torch.tensor([[.9, .2], [.1, .5]], dtype=torch.float64)
    direction = mobius_add(-p, v, c)
    theta = torch.acos((p * direction).sum(-1) / (p.norm(dim=-1) * direction.norm(dim=-1)))
    aperture = torch.asin(.1 * (1 - c * p.square().sum(-1)) / (c.sqrt() * p.norm(dim=-1)))
    expected = (theta - aperture).relu() - .01 * theta
    torch.testing.assert_close(cone_energy(v, p, c), expected, atol=1e-12, rtol=1e-12)


def test_inactive_hinge_keeps_outside_hinge_regularizer_gradient():
    c = torch.tensor(.5, dtype=torch.float64)
    p = torch.tensor([.6, 0.], dtype=torch.float64)
    v = torch.tensor([.9, .05], dtype=torch.float64, requires_grad=True)
    energy, stats = cone_energy(v, p, c, diagnostics=True)
    assert stats["hinge_active_fraction"] == 0
    energy.backward()
    assert v.grad.norm() > 0 and torch.isfinite(v.grad).all()


@pytest.mark.parametrize("parent,child", [([0., 0.], [0., 0.]), ([.3, .1], [.3, .1])])
def test_degenerate_angle_backward_is_finite(parent, child):
    p = torch.tensor(parent, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(child, dtype=torch.float64, requires_grad=True)
    cone_energy(v, p, torch.tensor(.5, dtype=torch.float64)).backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(v.grad).all()


def test_conditioning_preserves_weights_and_smooth_limit_preserves_radial_order():
    _, geometry = profile()
    model = FullPipeline(FakeBase(), "poincare", {**geometry, "dimension": 4})
    before = model.geometry.visual.weight.detach().clone()
    model.geometry.set_modality_norms(10., 12.)
    torch.testing.assert_close(model.geometry.visual.weight, before)
    x = torch.tensor([[.2, 0., 0., 0.], [.4, 0., 0., 0.], [1e4, 0., 0., 0.]],
                     dtype=torch.float64, requires_grad=True)
    mapped = model.geometry.map_tangent(x)
    radii = model.geometry.curvature.double().sqrt() * mapped.norm(dim=-1)
    assert (radii[1:] > radii[:-1]).all() and (radii <= .95 + 1e-12).all()
    mapped[:2].sum().backward()
    assert (x.grad[:2, 0] > 0).all()
    torch.testing.assert_close(mapped[:, 1:], torch.zeros_like(mapped[:, 1:]))
    state = model.downstream_state()
    restored = FullPipeline(FakeBase(), "poincare", {**geometry, "dimension": 4})
    restored.load_downstream(state)
    assert restored.geometry.visual_norm_scale == 10 and restored.geometry.text_norm_scale == 12
    with pytest.raises(ValueError, match="Conditioning norms"):
        restored.geometry.set_modality_norms(float("nan"), 1.)


def test_scalar_bias_norm_parameters_have_no_decay_and_frozen_curvature_is_excluded():
    cfg, geometry = profile()
    model = FullPipeline(FakeBase(), "poincare", {**geometry, "dimension": 4})
    optimizer = joint_optimizer(model, cfg)
    decay = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
    assert id(model.geometry.raw_c) not in decay and not model.geometry.raw_c.requires_grad
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert decay[id(p)] == (cfg["weight_decay"] if p.ndim >= 2 else 0.)
        else:
            assert id(p) not in decay


def test_single_temperature_cross_entropy_uses_same_logits_as_inference():
    cfg, geometry = profile()
    model = FullPipeline(FakeBase(), "poincare", {**geometry, "dimension": 4}).eval()
    x, lengths, target = inputs()
    _, c, a, aux = model.forward_with_aux(x, None, ["n", "a", "b"], lengths)
    _, parts = full_loss(c, a, target, lengths, aux, "poincare", {**cfg["loss"], "phi": .1})
    _, c2, a2 = model(x, None, ["n", "a", "b"], lengths)
    torch.testing.assert_close(a, a2)
    torch.testing.assert_close(c, c2)
    for i, n in enumerate(lengths):
        pooled = a2[i, :n].float().max(0).values
        expected = -(target[i] * pooled.log_softmax(-1)).sum()
        torch.testing.assert_close(parts["alignment"][i], expected)


def test_probe_selection_is_train_only_and_covers_available_classes():
    labels = torch.eye(4).repeat(3, 1)
    abnormal = (labels[:, 0] == 0).numpy()
    train = np.arange(8)
    selected = probe_indices(train, labels, abnormal, limit=6)
    assert set(selected) <= set(train) and len(set(selected)) == len(selected)
    assert (labels[selected].sum(0) > 0).all()
    assert np.array_equal(selected, probe_indices(train, labels, abnormal, limit=6))
    with pytest.raises(ValueError, match="Probe budget"):
        probe_indices(train, labels, abnormal, limit=3)


def test_health_distinguishes_clamp_risk_active_hinge_and_loss_gradients():
    _, geometry = profile()
    model = FullPipeline(FakeBase(), "poincare", {**geometry, "dimension": 4})
    x, lengths, target = inputs()
    _, c, a, aux = model.forward_with_aux(x, None, ["n", "a", "b"], lengths)
    _, parts = full_loss(c, a, target, lengths, aux, "poincare", LOSS)
    report = learning_health(model, c, a, aux, parts, target, lengths, LOSS)
    assert not report["loss_gradients"]["bce"]["text"]["connected"]
    assert report["loss_gradients"]["alignment"]["visual"]["norm"] > 0
    assert report["cone"]["pairs"] == 2
    assert [t["valid_steps"] for t in report["temporal"]] == [6, 3]
    assert report["microbatch_size"] == 2
    assert set(report["tangent_compression"]) == {"visual", "class_text", "fused_text"}
    assert all(p.grad is None for p in model.parameters())  # diagnostics did not accumulate grads
    full_loss(c, a, target, lengths, aux, "poincare", LOSS)[0].mean().backward()
    assert model.gradient_receipt()["prompt_tokens"]


def test_train_only_pilot_never_calls_evaluators_and_records_actual_budget(tmp_path):
    namespace = orchestrator_namespace()
    config, geometry = profile()
    original = copy.deepcopy(config)
    namespace.update(INPUT=tmp_path, WORK=tmp_path, DATASETS=("ucf", "xd"),
                     RUN_MODE="pilot", TRAIN_CONFIG=config, GEOMETRY_CONFIG=geometry)
    namespace["resolve_checkpoint"] = lambda *args: tmp_path / "original.pth"

    def forbidden(*args):
        raise AssertionError("Pilot must never evaluate official test or collect test reports")

    namespace["load_module"] = lambda *args: SimpleNamespace(
        collect_reports=forbidden, collect_latency=forbidden)
    calls = []

    def train(name, source, config, scripts):
        assert name in {"ucf_train", "xd_train"}
        assert config["TRAIN_CONFIG"]["epochs"] == 3 and not config["SMOKE_ONLY"]
        calls.append(name)
        output = config["OUTPUT"]
        output.mkdir(parents=True)
        (output / "training.json").write_text(json.dumps({"variants": {"v21": {
            "checkpoint_file": "v21/best.pth", "space": "poincare", "checkpoint_sha256": "fixture"}}}))
        return output / "training.json"

    namespace["run_dataset"] = train
    path = namespace["run_both"]()
    assert calls == ["ucf_train", "xd_train"] and config == original
    with ZipFile(path) as archive:
        summary = json.loads(archive.read("summary.json"))
    assert summary["mode"] == "pilot" and summary["train_config"]["epochs"] == 3
    assert not summary["branch_comparisons"]
    assert all(not record["reports"] for record in summary["runs"].values())
