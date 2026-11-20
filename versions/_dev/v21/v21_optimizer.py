"""Finite optimizer updates with bounded AMP retries of the identical effective batch."""

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def optimizer_update(model, optimizer, scaler, backward, *, amp, gradient_clip=None,
                     retry_limit=8, rng=None, before_step=None, diagnostics_path=None):
    """Retry overflow with lower GradScaler scale, unchanged data/RNG/buffers/weights.

    backward() performs forward/backward for the entire effective batch and returns
    detached log values. No optimizer update or data skipping occurs on overflow.
    Persistent nonfinite gradients raise after the bounded number of retries.
    """
    if type(retry_limit) is not int or retry_limit < 0:
        raise ValueError("amp retry_limit must be a nonnegative integer")
    if amp and not scaler.is_enabled():
        raise ValueError("AMP retries require an enabled GradScaler")
    if gradient_clip is not None and (not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError("gradient_clip must be finite and positive, or None")
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state, python_state = np.random.get_state(), random.getstate()
    generator_state = copy.deepcopy(rng.bit_generator.state) if rng is not None else None
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    events = []

    def write_diagnostics(status):
        if diagnostics_path is not None:
            path = Path(diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"status": status, "attempts": events}, indent=2),
                            encoding="utf-8")

    for attempt in range(retry_limit + 1):
        if attempt:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if rng is not None:
                rng.bit_generator.state = copy.deepcopy(generator_state)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
        optimizer.zero_grad(set_to_none=True)
        scale = scaler.get_scale()
        result = backward()
        scaler.unscale_(optimizer)
        named = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
        if not named:
            raise RuntimeError("Effective batch produced no gradients")
        flags = torch.stack([torch.isfinite(p.grad).all() for _, p in named])
        if not bool(flags.all()):
            bad = [name for (name, _), finite in zip(named, flags, strict=True) if not bool(finite)]
            events.append({"attempt": attempt + 1, "loss_scale": scale,
                           "nonfinite_parameters": bad})
            write_diagnostics("overflow" if amp else "failed_nonfinite_gradients")
            if not amp:
                raise RuntimeError(f"Non-finite unscaled gradients: {bad}")
            # GradScaler has already recorded nonfinite values during unscale_.
            # Its step MUST skip the real optimizer; update lowers the scale.
            scaler.step(optimizer)
            scaler.update()
            lower_scale = scaler.get_scale()
            if lower_scale >= scale or not math.isfinite(lower_scale) or lower_scale <= 0:
                raise RuntimeError("GradScaler did not reduce the overflowed loss scale")
            if attempt == retry_limit:
                write_diagnostics("failed_retry_limit")
                raise RuntimeError(
                    f"AMP could not produce finite gradients after {attempt + 1} attempts; "
                    f"last loss_scale={scale}, parameters={bad}; no update for this batch")
            print(f"AMP overflow: loss_scale {scale:g}->{lower_scale:g}; "
                  f"retrying identical batch ({attempt + 1}/{retry_limit}); groups={bad[:6]}",
                  flush=True)
            continue
        for name, parameter in named:
            if not parameter.requires_grad:
                raise RuntimeError(f"Frozen parameter received gradient: {name}")
        # Compute the diagnostic norm in FP64, without clipping nonfinite values.
        # In particular, max_norm=inf with norm=inf must never multiply grads by NaN.
        norm = torch.linalg.vector_norm(torch.stack([
            p.grad.detach().norm(dtype=torch.float64) for _, p in named]))
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Non-finite total norm despite finite individual gradients")
        if gradient_clip is not None:
            factor = (gradient_clip / norm.clamp_min(1e-12)).clamp(max=1.)
            for _, parameter in named:
                parameter.grad.mul_(factor.to(parameter.grad.dtype))
        if before_step is not None:
            before_step()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale:
            raise RuntimeError("Optimizer update unexpectedly skipped after finite-gradient checks")
        events.append({"attempt": attempt + 1, "loss_scale": scale,
                       "gradient_norm_before_clipping": float(norm), "nonfinite_parameters": []})
        write_diagnostics("updated")
        return result, {"overflow_retries": attempt, "loss_scale": scaler.get_scale(),
                        "gradient_norm": float(norm)}
    raise AssertionError("Unreachable optimizer retry state")
