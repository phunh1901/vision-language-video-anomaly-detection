"""Behavioral checks for freeze/unfreeze, real updates and phase checkpoints."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_v21 import GEOMETRY, LOSS, FakeBase, inputs
from train_v21 import fit, save_checkpoint
from v21_model import FullPipeline, full_loss
from v21_schedule import TwoStageTraining


def config():
    return {"epochs": 4, "learning_rate": 1e-4, "weight_decay": 5e-5,
            "gradient_clip": 1., "amp": False, "phi": {"ucf": .1, "xd": 1e-4},
            "loss": {k: v for k, v in LOSS.items() if k != "phi"},
            "two_stage": {"warmup_epochs": 2, "finetune_epochs": 2,
                          "head_lr": 1e-4, "finetune_head_lr": 5e-5,
                          "adapter_lr": 2e-5, "prompt_lr": 2e-5,
                          "curvature_lr": 1e-5, "scale_lr": 1e-5,
                              "finetune_curvature_lr": 5e-6, "finetune_scale_lr": 5e-6, "end_lr_ratio": .02}}


def test_actual_phase_updates_frozen_weights_and_adam_state():
    x, lengths, targets = inputs()
    model = FullPipeline(FakeBase(), "poincare", GEOMETRY)
    policy = TwoStageTraining(model, config())
    before = copy.deepcopy(model.state_dict())

    def update(epoch):
        policy.begin_epoch(epoch)
        policy.optimizer.zero_grad(set_to_none=True)
        _, c, a, aux = model.forward_with_aux(x, None, ["n", "a", "b"], lengths)
        loss, _ = full_loss(c, a, targets, lengths, aux, "poincare", LOSS)
        loss.mean().backward()
        receipt = policy.check_gradients()
        policy.optimizer.step()
        return receipt

    first = update(0)
    assert first["phase"] == "head_warmup"
    assert not model.base.training and model.geometry.training
    for name, value in model.base.state_dict().items():
        assert torch.equal(value, before["base." + name]), name
    assert not torch.equal(model.geometry.visual.weight, before["geometry.visual.weight"])
    step = policy.optimizer.state[model.geometry.visual.weight]["step"].item()
    warm = copy.deepcopy(model.state_dict())
    second = update(2)
    assert second["phase"] == "joint_finetune"
    assert model.base.training and not model.base.clipmodel.training
    assert policy.optimizer.state[model.geometry.visual.weight]["step"].item() == step + 1
    assert not torch.equal(model.base.temporal.weight, warm["base.temporal.weight"])
    assert not torch.equal(model.base.text_prompt.weight, warm["base.text_prompt.weight"])
    for name, value in model.base.clipmodel.state_dict().items():
        assert torch.equal(value, before["base.clipmodel." + name])
    policy.begin_epoch(3)
    rates = policy.receipt()["learning_rates"]
    assert rates["head"] == pytest.approx(1e-6)
    assert rates["adapter"] == pytest.approx(4e-7)
    assert all(group["weight_decay"] == 0 for group in policy.optimizer.param_groups
               if group["kind"] in {"curvature", "scale"})


@pytest.mark.parametrize("key,value", [
    ("warmup_epochs", 0), ("finetune_epochs", 1.5), ("head_lr", float("nan")),
    ("adapter_lr", -1), ("end_lr_ratio", 0), ("end_lr_ratio", float("inf")),
])
def test_invalid_policy_rejected_before_training(key, value):
    cfg = config()
    cfg["two_stage"][key] = value
    with pytest.raises(ValueError):
        TwoStageTraining(FullPipeline(FakeBase(), "poincare", GEOMETRY), cfg)


def test_tangent_clipping_preserves_different_radii_and_finite_gradients():
    model = FullPipeline(FakeBase(), "poincare", {**GEOMETRY, "max_tangent_norm": 3.})
    values = torch.tensor([[.1, 0., 0., 0.], [.2, 0., 0., 0.],
                           [10000., 1000., 0., 0.]], requires_grad=True)
    mapped = model.geometry.map_tangent(values)
    radius = model.geometry.curvature.sqrt() * mapped.norm(dim=-1)
    assert radius[0] < radius[1] < radius[2] < 1
    mapped.sum().backward()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("prefix", ["ucf", "xd"])
def test_cpu_smoke_crosses_both_phases_and_saves_recipe(tmp_path, prefix):
    torch.set_num_threads(2)

    class Pack:
        features = {str(i): i for i in range(8)}

        def array(self, index):
            return np.full((5, 512), .01 + .01 * index, np.float32)

    labels = torch.tensor([[1., 0., 0.] if i % 2 == 0 else [0., 1., 0.] for i in range(8)])
    result = fit(FakeBase(512), Pack(), [str(i) for i in range(8)], labels,
                 labels[:, 0].numpy() == 0, np.arange(4), np.arange(4, 8),
                 SimpleNamespace(visual_length=16), ["n", "a", "b"],
                 lambda lengths, width: torch.arange(width)[None] >= lengths[:, None],
                 True, config(), {**GEOMETRY, "dimension": 512, "max_tangent_norm": 3.},
                 tmp_path / "v21", 42, 1, torch.device("cpu"), prefix, "original")
    assert [row["training_phase"]["phase"] for row in result["history"]] == [
        "head_warmup", "joint_finetune"]
    assert result["training_policy"] == "two_stage_A"
    checkpoint = torch.load(tmp_path / result["checkpoint_file"], weights_only=True)
    assert checkpoint["train_config"]["two_stage"] == config()["two_stage"]
    state = torch.load(tmp_path / "v21" / "last_training_state.pt", weights_only=True)
    assert state["scheduler"]["phase"] == "joint_finetune"
    assert state["optimizer"]["state"]
    history = json.loads((tmp_path / "v21" / "history.json").read_text())
    assert len(history) == 2


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "best.pth"
    save_checkpoint({"epoch": 1}, destination)
    original = destination.read_bytes()

    def failed_save(payload, path):
        path.write_bytes(b"incomplete write")
        raise OSError("simulated disk full")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint({"epoch": 2}, destination)
    assert destination.read_bytes() == original
    assert not destination.with_name("best.pth.writing").exists()
