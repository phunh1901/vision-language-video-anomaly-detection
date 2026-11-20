"""Option B: prove fresh initialization, curvature curriculum, actual learning and reload."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_v21 import LOSS, FakeBase, inputs
from train_v21 import fit, initialize_base
from v21_model import FullPipeline, full_loss, load_full_model
from v21_schedule import FromScratchTraining

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

HERE = (REPOSITORY / 'versions/_dev/v21')


def settings():
    scope = {}
    exec((HERE / "notebook_config_B.py").read_text(encoding="utf-8"), scope)
    return copy.deepcopy(scope["TRAIN_CONFIG"]), copy.deepcopy(scope["GEOMETRY_CONFIG"])


def test_fresh_initialization_cannot_leak_v0_adapter_prompt_or_fusion():
    base = FakeBase()
    constructor = copy.deepcopy(base.state_dict())
    v0 = {name: torch.full_like(value, 7.) for name, value in constructor.items()}
    receipt = initialize_base(base, v0, "fresh_downstream")
    assert receipt["v0_downstream_loaded"] is False
    assert receipt["constructor_downstream_sha256"] == receipt["downstream_initial_sha256"]
    for name, actual in base.state_dict().items():
        assert torch.equal(actual, v0[name] if name.startswith("clipmodel.") else constructor[name])


def test_initialization_rejects_incomplete_clip_and_unknown_mode():
    base = FakeBase()
    with pytest.raises(ValueError, match="incomplete"):
        initialize_base(base, {}, "fresh_downstream")
    with pytest.raises(ValueError, match="Unknown initialization"):
        initialize_base(base, base.state_dict(), "silently_reuse_old_checkpoint")


def test_curvature_and_scales_fixed_then_learn_with_backbone_active():
    torch.set_num_threads(2)
    x, lengths, targets = inputs()
    cfg, geometry = settings()
    model = FullPipeline(FakeBase(), "poincare", {**geometry, "dimension": 4})
    policy = FromScratchTraining(model, cfg)
    original = copy.deepcopy(model.state_dict())

    def update(epoch):
        policy.begin_epoch(epoch)
        policy.optimizer.zero_grad(set_to_none=True)
        _, c, a, aux = model.forward_with_aux(x, None, ["n", "a", "b"], lengths)
        loss, _ = full_loss(c, a, targets, lengths, aux, "poincare", LOSS)
        loss.mean().backward()
        policy.check_gradients()
        policy.optimizer.step()

    update(0)
    assert model.base.training and not model.base.clipmodel.training
    assert model.geometry.curvature.item() == pytest.approx(.05)
    for name in ("raw_c", "raw_v", "raw_t"):
        parameter = getattr(model.geometry, name)
        assert parameter.grad is None
        assert torch.equal(parameter, original["geometry." + name])
    assert not torch.equal(model.base.temporal.weight, original["base.temporal.weight"])
    assert not torch.equal(model.base.text_prompt.weight, original["base.text_prompt.weight"])
    update(3)
    for name in ("raw_c", "raw_v", "raw_t"):
        parameter = getattr(model.geometry, name)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert not torch.equal(parameter, original["geometry." + name])
    for name, actual in model.base.clipmodel.state_dict().items():
        assert torch.equal(actual, original["base.clipmodel." + name])
    policy.begin_epoch(29)
    rates = policy.receipt()["learning_rates"]
    assert rates["head"] == pytest.approx(1e-6)
    assert rates["curvature"] == pytest.approx(1e-7)
    assert rates["scale"] == pytest.approx(1e-7)


@pytest.mark.parametrize("curvature", [.01, .05, 1.99])
def test_extended_curvature_bounds_and_gradients(curvature):
    _, cfg = settings()
    model = FullPipeline(FakeBase(), "poincare", {**cfg, "dimension": 4, "curvature": curvature})
    assert model.geometry.curvature.item() == pytest.approx(curvature, rel=1e-5)
    mapped = model.geometry.map_visual(torch.randn(2, 4))
    mapped.square().sum().backward()
    assert torch.isfinite(model.geometry.raw_c.grad)
    assert (model.geometry.curvature * mapped.square().sum(-1) < 1).all()


@pytest.mark.parametrize("prefix", ["ucf", "xd"])
def test_b_smoke_fit_both_phases_and_exact_checkpoint_reload(tmp_path, prefix):
    cfg, geometry = settings()
    torch.set_num_threads(2)

    class Pack:
        features = {str(i): i for i in range(8)}

        def array(self, index):
            return np.full((5, 512), .01 + index * .01, np.float32)

    base = FakeBase(512)
    base_state = copy.deepcopy(base.state_dict())
    labels = torch.tensor([[1., 0., 0.] if i % 2 == 0 else [0., 1., 0.] for i in range(8)])
    result = fit(base, Pack(), [str(i) for i in range(8)], labels,
                 labels[:, 0].numpy() == 0, np.arange(4), np.arange(4, 8),
                 SimpleNamespace(visual_length=16), ["n", "a", "b"],
                 lambda lengths, width: torch.arange(width)[None] >= lengths[:, None],
                 True, cfg, geometry, tmp_path / "v21", 42, 1, torch.device("cpu"), prefix, "v0")
    assert result["training_policy"] == "from_scratch_B"
    assert [row["training_phase"]["phase"] for row in result["history"]] == [
        "low_curvature_warmup", "hyperbolic_joint"]
    assert result["preflight"]["classifier_bias_calibrated"]
    assert result["preflight"]["classifier_bias"] > 0
    assert result["preflight"]["geometry_probes"]
    checkpoint = tmp_path / result["checkpoint_file"]
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["train_config"]["initialization"] == "fresh_downstream"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    first, second = FakeBase(512), FakeBase(512)
    first.load_state_dict(base_state)
    second.load_state_dict(base_state)
    a, _ = load_full_model(first, "poincare", geometry, checkpoint, digest, "v0", "cpu")
    b, _ = load_full_model(second, "poincare", geometry, checkpoint, digest, "v0", "cpu")
    values, lengths = torch.randn(1, 5, 512), torch.tensor([5])
    with torch.no_grad():
        left = a(values, None, ["n", "a", "b"], lengths)
        right = b(values, None, ["n", "a", "b"], lengths)
    for one, two in zip(left, right, strict=True):
        torch.testing.assert_close(one, two)


def test_standalone_b_recipe_sources_and_runner_are_consistent():
    notebook = json.loads((HERE / "fixtures/recipe_notebooks/run_vadclip_v21_B.ipynb.json").read_text(encoding="utf-8"))
    scope = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"B_cell{index}", "exec")
    for index in (1, 2):
        exec("".join(notebook["cells"][index]["source"]), scope)
    assert scope["TRAIN_CONFIG"]["initialization"] == "fresh_downstream"
    assert "two_stage" not in scope["TRAIN_CONFIG"]
    assert scope["TRAIN_CONFIG"]["epochs"] == 30
    assert scope["INCLUDE_PREDICTIONS"]
    assert set(scope["DATASETS"]) == {"ucf", "xd"}
    for name, source in scope["SOURCES"].items():
        compile(source, name, "exec")
        if (HERE / name).is_file():
            assert source == (HERE / name).read_text(encoding="utf-8")
    assert '"training_recipe"' in scope["SOURCES"]["orchestrator_v21.py"]
