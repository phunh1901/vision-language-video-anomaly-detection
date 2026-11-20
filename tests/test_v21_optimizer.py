"""Real CPU GradScaler tests emulate FP16 overflow; no CUDA/GPU results claimed."""

import copy
import json

import numpy as np
import pytest
import torch
from torch import nn
from v21_optimizer import optimizer_update


class HalfGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return values.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient.half().float()


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.update_count = 0

    def step(self, closure=None):
        self.update_count += 1
        return super().step(closure)


def test_fp16_scaled_overflow_recovers_same_batch_with_one_real_update(tmp_path):
    model = nn.ParameterDict({"p": nn.Parameter(torch.tensor(2.))})
    optimizer = CountingSGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024)

    def backward():
        loss = HalfGradient.apply(model["p"]) * 1000
        scaler.scale(loss).backward()
        return float(loss.detach())

    _, info = optimizer_update(model, optimizer, scaler, backward, amp=True, retry_limit=8,
                               diagnostics_path=tmp_path / "gradient.json")
    assert info["overflow_retries"] == 4
    assert optimizer.update_count == 1 and scaler.get_scale() == 64
    torch.testing.assert_close(model["p"], torch.tensor(-98.))
    report = json.loads((tmp_path / "gradient.json").read_text())
    assert report["status"] == "updated" and len(report["attempts"]) == 5
    assert report["attempts"][0]["nonfinite_parameters"] == ["p"]


def test_persistent_nan_stops_without_optimizer_update(tmp_path):
    model = nn.ParameterDict({"p": nn.Parameter(torch.tensor(2.))})
    optimizer = CountingSGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1.)

    def backward():
        scaler.scale(model["p"] * torch.tensor(float("nan"))).backward()

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        optimizer_update(model, optimizer, scaler, backward, amp=True, retry_limit=2,
                         diagnostics_path=tmp_path / "gradient.json")
    assert optimizer.update_count == 0 and model["p"].item() == 2
    report = json.loads((tmp_path / "gradient.json").read_text())
    assert report["status"] == "failed_retry_limit"


def test_finite_large_gradient_is_not_poisoned_by_infinite_clip_threshold():
    model = nn.ParameterDict({"p": nn.Parameter(torch.tensor([2., 3.]))})
    optimizer = CountingSGD(model.parameters(), lr=1e-30)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def backward():
        scaler.scale(model["p"].sum() * 1e30).backward()

    _, info = optimizer_update(model, optimizer, scaler, backward, amp=False, gradient_clip=None)
    assert np.isfinite(info["gradient_norm"])
    assert torch.isfinite(model["p"]).all()
    torch.testing.assert_close(model["p"], torch.tensor([1., 2.]))


def test_retry_restores_dropout_numpy_sampling_and_running_buffers():
    torch.manual_seed(4)
    initial = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Dropout(.4), nn.Linear(3, 1))
    original = copy.deepcopy(initial.state_dict())
    states, traces = [], []
    for inject_failure in (False, True):
        model = copy.deepcopy(initial)
        model.load_state_dict(original)
        optimizer = CountingSGD(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=1.)
        rng = np.random.default_rng(42)
        torch.manual_seed(61)
        observed, calls = [], [0]

        def hook(grad):
            calls[0] += 1
            return grad * float("inf") if inject_failure and calls[0] == 1 else grad

        next(model.parameters()).register_hook(hook)

        def backward():
            sample = torch.tensor(rng.normal(size=(4, 3)), dtype=torch.float32)
            output = model(sample)
            observed.append((sample.detach().clone(), output.detach().clone()))
            scaler.scale(output.square().mean()).backward()

        optimizer_update(model, optimizer, scaler, backward, amp=True, retry_limit=2, rng=rng)
        assert optimizer.update_count == 1
        states.append(copy.deepcopy(model.state_dict()))
        traces.append(observed)
    assert len(traces[1]) == 2
    for first, repeated in zip(traces[1][0], traces[1][1], strict=True):
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name])


def test_disabled_amp_fails_immediately_on_nonfinite_gradients():
    model = nn.Linear(1, 1)
    optimizer = CountingSGD(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def backward():
        (model.weight.sum() * float("inf")).backward()

    with pytest.raises(RuntimeError, match="Non-finite unscaled gradients"):
        optimizer_update(model, optimizer, scaler, backward, amp=False)
    assert optimizer.update_count == 0
