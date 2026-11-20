"""Behavioral architecture/precision tests. Fixtures are NOT the author's CLIP/GCN code."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from train_v21 import fit
from v21_flow import EncoderFlowAudit
from v21_geometry import expmap0, logmap0, poincare_distance
from v21_model import FullPipeline, full_loss

GEOMETRY = {"dimension": 8, "curvature": .5, "scale": .7, "temperature": .07,
            "text_projection": False, "geometry_dtype": "float64",
            "encoder_precision": "float32"}
LOSS = {"psi": 1e-3, "phi": .1, "cone_aperture_scale": .1,
        "entail_regularizer": .01, "separation_margin": 1.}


@pytest.mark.parametrize("curvature", [.1, .5, 1.])
def test_fp64_roundtrip_does_not_clip_moderate_large_norm_or_kill_radial_gradient(curvature):
    # Previous guard clipped norm 10 to 8.63099 at c=.5 and killed radial gradients.
    # Curvature's parameter is FP32, so sqrt(c) must also be evaluated after casting.
    c = torch.tensor(curvature, requires_grad=True)
    v = torch.tensor([[10., 0.]], dtype=torch.float64, requires_grad=True)
    x = expmap0(v, c)
    restored = logmap0(x, c)
    torch.testing.assert_close(restored, v, rtol=3e-8, atol=1e-9)
    restored.sum().backward()
    torch.testing.assert_close(v.grad, torch.ones_like(v), rtol=1e-7, atol=1e-7)
    assert torch.isfinite(c.grad)
    torch.testing.assert_close(poincare_distance(x, torch.zeros_like(x), c),
                               2 * v.norm(dim=-1), rtol=4e-8, atol=1e-8)
    assert (c.double() * x.square().sum(-1) < 1).all()


class FlowBase(nn.Module):
    """Trainable local attention + two global graph paths + prompted frozen text fixture."""

    def __init__(self):
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(8, 2, 16, dropout=0., batch_first=True)
        self.gc1, self.gc2 = nn.Linear(8, 4), nn.Linear(4, 4)
        self.gc3, self.gc4 = nn.Linear(8, 4), nn.Linear(4, 4)
        self.linear = nn.Linear(8, 8)
        self.frame_position_embeddings = nn.Embedding(8, 8)
        self.text_prompt_embeddings = nn.Embedding(3, 8)
        self.clipmodel = nn.Module()
        self.clipmodel.visual = nn.Linear(8, 8)
        self.clipmodel.transformer = nn.TransformerEncoderLayer(
            8, 2, 16, dropout=0., batch_first=True)
        self.clipmodel.half()  # emulate a loader which explicitly stores CLIP in FP16
        self.mlp1 = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 8))
        self.classifier, self.mlp2 = nn.Linear(8, 1), nn.Linear(8, 8)

    def encode_video(self, images, padding_mask, lengths):
        position = self.frame_position_embeddings(torch.arange(images.shape[1]))
        local_mask = (torch.arange(images.shape[1])[:, None] // 2
                      != torch.arange(images.shape[1])[None, :] // 2)
        x = self.temporal(images + position, src_mask=local_mask)
        similarity = (x @ x.transpose(-1, -2)).softmax(-1)
        time = torch.arange(images.shape[1]).float()
        adjacency = (-(time[:, None] - time[None]).abs()).softmax(-1)
        first = self.gc2((similarity @ self.gc1(x)).tanh())
        second = self.gc4((adjacency @ self.gc3(x)).tanh())
        return self.linear(torch.cat((first, second), -1))

    def encode_textprompt(self, text):
        prompt = self.text_prompt_embeddings(torch.arange(len(text)))
        dtype = next(self.clipmodel.parameters()).dtype
        return self.clipmodel.transformer(prompt.to(dtype)[None])[0]


def probe(model):
    images = torch.randn(2, 6, 8)
    lengths = torch.tensor([6, 3])
    targets = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    with EncoderFlowAudit(model) as audit:
        _, c, a, aux = model.forward_with_aux(images, None, ["n", "a", "b"], lengths)
        loss, _ = full_loss(c, a, targets, lengths, aux, "poincare", LOSS)
        loss.mean().backward()
        return audit.receipt()


def test_observed_order_and_separate_local_global_text_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = FullPipeline(FlowBase(), "poincare", GEOMETRY).train()
    receipt = probe(model)
    assert not receipt["image_encoder_called"] and not receipt["frozen_clip_gradients"]
    assert all(receipt["nonzero_finite_gradient_groups"].values())
    assert set(receipt["nonzero_finite_gradient_groups"]) >= {
        "local_transformer", "global_gc1", "global_gc2", "global_gc3", "global_gc4",
        "visual_projection", "text_prompts"}
    assert "encode_video" in receipt["encoder_source"]
    model.zero_grad(set_to_none=True)
    assert not any(module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("fault", ["bypass_local", "bypass_global", "detach_temporal", "detach_text",
                                 "image_encoder"])
def test_flow_audit_rejects_missing_or_disconnected_real_stages(fault):
    model = FullPipeline(FlowBase(), "poincare", GEOMETRY).train()
    base = model.base
    encode_video = base.encode_video
    if fault == "bypass_local":
        base.temporal.forward = lambda images, **kwargs: images
        # Module hooks still run, but the Transformer parameters must have gradients.
    elif fault == "bypass_global":
        base.encode_video = lambda images, mask, lengths: base.linear(images)
    elif fault == "detach_temporal":
        base.encode_video = lambda *args: encode_video(*args).detach()
    elif fault == "detach_text":
        encode_text = base.encode_textprompt
        base.encode_textprompt = lambda *args: encode_text(*args).detach()
    else:
        base.encode_video = lambda images, mask, lengths: encode_video(
            base.clipmodel.visual(images), mask, lengths)
    with pytest.raises(RuntimeError, match="Disconnected|Broken|unexpectedly"):
        probe(model)
    assert not any(module._forward_hooks for module in model.modules())


def test_fp32_encoder_policy_undoes_loaded_half_weights_and_external_autocast():
    torch.manual_seed(8)
    model = FullPipeline(FlowBase(), "poincare", GEOMETRY).train()
    assert all(p.dtype == torch.float32 for p in model.base.clipmodel.parameters())
    images, lengths = torch.randn(2, 6, 8), torch.tensor([6, 3])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with EncoderFlowAudit(model) as audit:
            _, c, a, _ = model.forward_with_aux(images, None, ["n", "a", "b"], lengths)
    for event in audit.events:
        assert event["output"]["dtype"] == "torch.float32"
    _, c_reference, a_reference, _ = model.forward_with_aux(images, None, ["n", "a", "b"], lengths)
    torch.testing.assert_close(c, c_reference, rtol=0, atol=0)
    torch.testing.assert_close(a, a_reference, rtol=0, atol=0)


def test_train_calls_flow_guard_and_writes_observed_evidence(tmp_path):
    from test_v21_paper import profile

    config, _ = profile()
    config.update(epochs=1, microbatch_size=4)

    class Pack:
        features = {str(i): i for i in range(4)}

        def array(self, index):
            # Real trainer expects stored ViT-B/16 dimension. Slice at fixture boundary.
            return np.full((6, 512), .01 + .01 * index, np.float32)

    class StoredFeaturesBase(FlowBase):
        def encode_video(self, images, padding_mask, lengths):
            return super().encode_video(images[..., :8], padding_mask, lengths)

    base = StoredFeaturesBase()
    labels = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    before = copy.deepcopy(base.temporal.state_dict())
    fit(base, Pack(), list(Pack.features), labels, labels[:, 0].numpy() == 0,
        np.arange(4), np.array([], dtype=int), SimpleNamespace(visual_length=8),
        ["n", "a", "b"], lambda lengths, width: torch.arange(width)[None] >= lengths[:, None],
        False, config, GEOMETRY, tmp_path / "v21", 42, 2, torch.device("cpu"), "ucf", "v0")
    assert (tmp_path / "v21" / "encoder_flow.json").is_file()
    assert any(not torch.equal(before[name], value) for name, value in base.temporal.state_dict().items())
