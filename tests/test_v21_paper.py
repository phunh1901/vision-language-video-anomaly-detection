"""Published-equation/profile checks; synthetic CPU fixtures are not author benchmarks."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import v21_data as data
from test_v21 import GEOMETRY, LOSS, FakeBase, inputs
from train_v21 import fit
from v21_geometry import expmap0
from v21_model import FullPipeline, full_loss

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

HERE = (REPOSITORY / 'versions/_dev/v21')


def profile():
    scope = {}
    exec((HERE / "notebook_config_PAPER.py").read_text(encoding="utf-8"), scope)
    return copy.deepcopy(scope["TRAIN_CONFIG"]), copy.deepcopy(scope["GEOMETRY_CONFIG"])


def test_eq6_direct_text_mapping_has_no_added_linear():
    _, geometry = profile()
    model = FullPipeline(FakeBase(), "poincare", {**geometry, "dimension": 4})
    text = torch.randn(3, 4, requires_grad=True)
    actual = model.geometry.map_text(text)
    tangent = (torch.nn.functional.softplus(model.geometry.raw_t) * text).double()
    limit = math.atanh(geometry["max_relative_radius"]) / model.geometry.curvature.double().sqrt()
    z = tangent.norm(dim=-1, keepdim=True) / limit
    expected = expmap0(tangent * z.tanh() / z, model.geometry.curvature)
    torch.testing.assert_close(actual, expected)
    assert not any(name.startswith("geometry.text.") for name in model.state_dict())
    actual.sum().backward()
    assert text.grad is not None and torch.isfinite(text.grad).all()


def test_literal_eq10_temperature_applied_after_pooling():
    images, lengths, targets = inputs()
    model = FullPipeline(FakeBase(), "poincare", GEOMETRY)
    _, c, a, aux = model.forward_with_aux(images, None, ["n", "a", "b"], lengths)
    _, parts = full_loss(c, a, targets, lengths, aux, "poincare",
                         {**LOSS, "alignment_loss_temperature": .07})
    for i, length in enumerate(lengths):
        pooled = a[i, :length].max(dim=0).values  # both fixture videos have k=1
        expected = -(targets[i] * torch.log_softmax(pooled / .07, -1)).sum()
        torch.testing.assert_close(parts["alignment"][i], expected)


def test_official_mode_keeps_all_training_clips_and_discloses_source_overlap():
    rows = [{"path": f"movie{i}__#clip__0.npy", "label": "A" if i % 2 else "B"}
            for i in range(10)]
    test = [{"path": "movie0__#different__0.npy", "label": "B"}]
    features = {data.video_id(row["path"]): {} for row in rows}

    def encode(labels):
        return torch.tensor([[1., 0.] if label == "A" else [0., 1.] for label in labels])

    _, _, _, _, train, dev, excluded, overlap = data.dataset_split(
        rows, test, features, encode, "xd", 42, "official_train_final_epoch")
    assert train.tolist() == list(range(10))
    assert not len(dev) and not len(excluded) and overlap == 1
    with pytest.raises(ValueError, match="ID official train"):
        data.dataset_split(rows, rows[:1], features, encode, "xd", 42, "official_train_final_epoch")


def test_physical_batch_matches_accumulation_and_fixed_final_checkpoint(tmp_path):
    torch.set_num_threads(2)
    cfg, geometry = profile()
    # This test isolates accumulation/selection using FakeBase, which has no GCN.
    # The enabled production guard is exercised with local/global modules in test_v21_flow.
    cfg.update(epochs=2, amp=False, verify_encoder_flow=False)
    geometry["dimension"] = 4

    class SmallBase(FakeBase):
        # Keep the real trainer's cached-feature [T,512] I/O, but use a small
        # downstream fixture for an exact parameter-level accumulation check.
        # The separate physical-batch test exercises the full 512-d contract.
        def encode_video(self, images, padding_mask, lengths):
            return super().encode_video(images[..., :4], padding_mask, lengths)

    class Pack:
        features = {str(i): i for i in range(4)}

        def array(self, index):
            # Distinct time/channel values avoid tied max-pooling subgradients.
            rng = np.random.default_rng(100 + index)
            return rng.normal(.01 + .01 * index, .02, (5, 512)).astype(np.float32)

    torch.manual_seed(3)
    initial = SmallBase(4).state_dict()
    labels = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 1., 0.]])
    checkpoints = []
    for microbatch in (1, 4):
        base = SmallBase(4)
        base.load_state_dict(initial)
        result = fit(base, Pack(), [str(i) for i in range(4)], labels,
                     labels[:, 0].numpy() == 0, np.arange(4), np.array([], dtype=int),
                     SimpleNamespace(visual_length=16), ["n", "a", "b"],
                     lambda lengths, width: torch.arange(width)[None] >= lengths[:, None],
                     False, {**cfg, "microbatch_size": microbatch}, geometry,
                     tmp_path / str(microbatch) / "v21", 42, 2, torch.device("cpu"), "ucf", "v0")
        assert result["best_epoch"] == 2
        assert result["best_dev_primary_video_ap"] is None
        assert result["selection_branch"] == "fixed_final_epoch"
        assert all(row["dev_classification_video_ap"] is None for row in result["history"])
        checkpoints.append(torch.load(tmp_path / str(microbatch) / result["checkpoint_file"],
                                      weights_only=True)["downstream"])
    for name in checkpoints[0]:
        torch.testing.assert_close(checkpoints[0][name], checkpoints[1][name], atol=2e-6,
                                   rtol=2e-4, msg=lambda detail: f"{name}: {detail}")


def test_standalone_sanity_settings_and_disclosure():
    notebook = json.loads((HERE / "fixtures/recipe_notebooks/run_vadclip_v21_PAPER.ipynb.json").read_text(encoding="utf-8"))
    scope = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"paper_cell{index}", "exec")
    for index in (1, 2):
        exec("".join(notebook["cells"][index]["source"]), scope)
    cfg, geom = scope["TRAIN_CONFIG"], scope["GEOMETRY_CONFIG"]
    assert cfg["epochs"] == 30 and cfg["learning_rate"] == 5e-4
    assert cfg["microbatch_size"] == scope["HALF_BATCH"] * 2 == 64
    assert not cfg["calibrate_initial_scale"] and cfg["calibrate_classifier_bias"]
    assert cfg["condition_modality_norms"] and cfg["no_decay_scalars_bias_norm"]
    assert cfg["single_temperature"] and cfg["loss"]["alignment_loss_temperature"] == 1.
    assert cfg["gradient_clip"] == 1. and geom["max_tangent_norm"] is None
    assert not geom["curvature_trainable"] and geom["max_relative_radius"] == .95
    assert cfg["sanity_fix"] and not cfg["paper_design"]
    assert not geom["text_projection"]
    assert geom["curvature_min"] == .1 and geom["curvature_max"] == 1.
    assert not cfg["reproduction_disclosure"]["exact_author_code_verified"]
    assert cfg["verify_encoder_flow"] and not cfg["amp"]
    assert geom["encoder_precision"] == "float32"
    assert "two_stage" not in cfg and "from_scratch" not in cfg
    for name, source in scope["SOURCES"].items():
        compile(source, name, "exec")
        if (HERE / name).is_file():
            assert source == (HERE / name).read_text(encoding="utf-8")


def test_paper_smoke_uses_a_real_batch_of_64(tmp_path):
    torch.set_num_threads(2)
    cfg, geometry = profile()
    cfg["amp"] = False  # CPU fixture, not a GPU resource estimate.
    cfg["verify_encoder_flow"] = False  # FakeBase tests physical batch size; no GCN fixture.

    class TrackingBase(FakeBase):
        seen_batches = []

        def encode_video(self, images, padding_mask, lengths):
            self.seen_batches.append(images.shape[0])
            return super().encode_video(images, padding_mask, lengths)

    class Pack:
        features = {str(i): i for i in range(64)}

        def array(self, index):
            return np.full((5, 512), .01 + .0001 * index, np.float32)

    base = TrackingBase(512)
    labels = torch.tensor([[1., 0., 0.] if i < 32 else [0., 1., 0.] for i in range(64)])
    result = fit(base, Pack(), [str(i) for i in range(64)], labels,
                 labels[:, 0].numpy() == 0, np.arange(64), np.array([], dtype=int),
                 SimpleNamespace(visual_length=16), ["n", "a", "b"],
                 lambda lengths, width: torch.arange(width)[None] >= lengths[:, None],
                 True, cfg, geometry, tmp_path / "v21", 42, 32, torch.device("cpu"), "xd", "v0")
    assert 64 in base.seen_batches
    assert result["microbatch"] == 64 and result["effective_batch"] == 64
    assert result["best_epoch"] == 1  # smoke fixed epoch, never called a full benchmark
