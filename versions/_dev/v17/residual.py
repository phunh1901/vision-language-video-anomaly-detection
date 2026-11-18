"""Bounded additive residual from the binary C logit to abnormal A logits."""

import math


def residual_logits(logits_c, logits_a, eta):
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("RESIDUAL_ETA must be finite and in [0, 1].")
    if logits_a.shape[:-1] != logits_c.shape[:-1] or logits_c.shape[-1] != 1:
        raise ValueError("C and A logits must use the same block/snippet grid.")
    if logits_a.shape[-1] < 2:
        raise ValueError("A logits require Normal at index 0 and at least one abnormal class.")
    if eta == 0:
        return logits_a
    delta = eta * (2 * logits_c.detach().float().sigmoid() - 1)
    adjusted = logits_a.clone()
    adjusted[..., 1:] = (adjusted[..., 1:].float() + delta).to(logits_a.dtype)
    return adjusted
