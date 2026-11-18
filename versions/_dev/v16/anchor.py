"""Frozen, offline per-video Normal anchor. No labels enter this module.

Native inputs: normalized visual [blocks, width, D], normalized text [blocks, D, C],
binary logits [blocks, width, 1], alignment logits [blocks, width, C].
All blocks MUST belong to one video. Padding is excluded by lengths.
Only the Normal logit (column 0) is changed; native C and other A logits are retained.
"""

import ast
import hashlib
import math

import torch
import torch.nn.functional as F

DEFAULT_CONFIG = {
    "beta_max": 0.10,
    "normal_c_max": 0.20,
    "normal_a_max": 0.20,
    "bottom_fraction": 0.20,
    "min_snippets": 8,
    "min_coherence": 0.50,
    "neighbor_radius": 1,
}


def validate_config(cfg):
    if set(cfg) != set(DEFAULT_CONFIG):
        raise ValueError("Unexpected/missing anchor configuration keys")
    for key in ("beta_max", "normal_c_max", "normal_a_max", "bottom_fraction", "min_coherence"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f"Invalid finite number: {key}")
    if not (0 <= cfg["beta_max"] <= 1 and 0 < cfg["bottom_fraction"] <= 1):
        raise ValueError("Invalid beta or bottom fraction")
    if not all(0 < cfg[k] < 1 for k in ("normal_c_max", "normal_a_max", "min_coherence")):
        raise ValueError("Confidence/coherence cutoffs must be in (0, 1)")
    if type(cfg["min_snippets"]) is not int or cfg["min_snippets"] < 1:
        raise ValueError("min_snippets must be a positive integer")
    if type(cfg["neighbor_radius"]) is not int or cfg["neighbor_radius"] < 0:
        raise ValueError("neighbor_radius must be a nonnegative integer")


def instrument_source(source):
    """Capture the exact operands of native A matmul; fail closed on source drift.

No reimplementation of LGT, visual prompting, text encoder or logits. Parsing
is restricted to the pinned author's source, never a checkpoint/data payload.
"""
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CLIPVAD"]
    if len(classes) != 1:
        raise ValueError("Expected one upstream CLIPVAD class")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    if len(methods) != 1:
        raise ValueError("Expected one upstream forward")
    method = methods[0]
    returns = [n for n in method.body if isinstance(n, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Tuple):
        raise ValueError("Native forward return structure changed")
    result = returns[0]
    if len(result.value.elts) != 3 or not isinstance(result.value.elts[2], ast.Name):
        raise ValueError("Expected native (text, C logits, A logits)")
    target = result.value.elts[2].id
    assignments = [n for n in method.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError("Cannot identify native alignment computation uniquely")
    expr = assignments[0].value
    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div)
            and isinstance(expr.left, ast.BinOp) and isinstance(expr.left.op, ast.MatMult)
            and isinstance(expr.right, ast.Constant)
            and isinstance(expr.right.value, (int, float)) and expr.right.value > 0):
        raise ValueError("Expected native visual @ text / positive literal temperature")
    if not isinstance(expr.left.left, ast.Name):
        raise ValueError("Unexpected visual operand; inspect pinned source")
    # Permit native tensor dtype conversion, but no arbitrary function calls.
    right = expr.left.right
    if not (isinstance(right, ast.Name) or (
        isinstance(right, ast.Call) and isinstance(right.func, ast.Attribute)
        and isinstance(right.func.value, ast.Name) and right.func.attr == "type"
    )):
        raise ValueError("Unexpected text operand; inspect pinned source")
    capture = ast.parse(
        "self._v16_capture = ("
        f"{ast.unparse(expr.left.left)}, {ast.unparse(right)}, {expr.right.value!r})"
    ).body[0]
    method.body.insert(method.body.index(result), capture)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def patch_model_file(path):
    source = path.read_text(encoding="utf-8")
    patched = instrument_source(source)
    path.write_text(patched, encoding="utf-8")
    return {"original_model_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "instrumented_model_sha256": hashlib.sha256(patched.encode()).hexdigest()}


@torch.no_grad()
def adapt_normal(visual, text, logits_c, logits_a, lengths, cfg):
    validate_config(cfg)
    visual, text = visual.detach(), text.detach()
    logits_c, logits_a = logits_c.detach(), logits_a.detach()
    if visual.ndim != 3 or text.ndim != 3 or logits_a.ndim != 3:
        raise ValueError("Expected block-major 3D tensors")
    blocks, width, dim = visual.shape
    classes = logits_a.shape[-1]
    if (text.shape != (blocks, dim, classes) or logits_a.shape[:2] != (blocks, width)
            or logits_c.shape != (blocks, width, 1) or classes < 2):
        raise ValueError("Native embedding/logit shapes do not match")
    sizes = torch.as_tensor(lengths, device=visual.device).reshape(-1)
    if len(sizes) != blocks or (sizes < 0).any() or (sizes > width).any():
        raise ValueError("Invalid native block lengths")
    valid = torch.arange(width, device=visual.device)[None, :] < sizes[:, None]
    if not valid.any() or any(not x.isfinite().all() for x in (visual, text, logits_c, logits_a)):
        raise ValueError("Empty video or nonfinite native tensors")
    v = visual[valid].float()
    pc = logits_c[..., 0][valid].float().sigmoid()
    pa = 1 - logits_a[valid].float().softmax(-1)[:, 0]
    confidence = torch.maximum(pc, pa)
    eligible = (pc <= cfg["normal_c_max"]) & (pa <= cfg["normal_a_max"])
    # Require stable low scores in real adjacent snippets, including across blocks.
    stable = eligible.clone()
    for shift in range(1, cfg["neighbor_radius"] + 1):
        stable[shift:] &= eligible[:-shift]
        stable[:-shift] &= eligible[shift:]
    indices = stable.nonzero().flatten()
    cap = max(1, math.ceil(len(v) * cfg["bottom_fraction"]))
    indices = indices[torch.argsort(confidence[indices], stable=True)[:cap]]
    info = {"steps": len(v), "eligible_count": int(stable.sum()),
            "selected_count": len(indices), "beta": 0.0, "coherence": None,
            "fallback": None, "selected_indices": indices.cpu().tolist(),
            "mean_abs_normal_logit_delta": 0.0}
    if cfg["beta_max"] == 0:
        info["fallback"] = "beta_zero_control"
        return logits_a, info, None
    if len(indices) < cfg["min_snippets"]:
        info["fallback"] = "insufficient_confident_snippets"
        return logits_a, info, None
    weights = (1 - pc[indices]) * (1 - pa[indices])
    pooled = (v[indices] * weights[:, None]).sum(0) / weights.sum()
    norm = pooled.norm()
    if norm <= 1e-8:
        info["fallback"] = "degenerate_anchor"
        return logits_a, info, None
    anchor = F.normalize(pooled, dim=0)
    coherence = float((v[indices] @ anchor).mean())
    info["coherence"] = coherence
    if coherence < cfg["min_coherence"]:
        info["fallback"] = "incoherent_selected_features"
        return logits_a, info, None
    certainty = float((1 - confidence[indices]).mean())
    support = min(1.0, len(indices) / max(cfg["min_snippets"], cap))
    beta = cfg["beta_max"] * certainty * support * max(0.0, min(1.0, coherence))
    native_normal = text[:, :, 0].float()
    mixed = (1 - beta) * native_normal + beta * anchor[None, :]
    if (mixed.norm(dim=-1) <= 1e-8).any():
        info["fallback"] = "degenerate_mixture"
        return logits_a, info, None
    normal = F.normalize(mixed, dim=-1)
    info["beta"] = beta
    return logits_a, info, normal


@torch.no_grad()
def anchor_forward(model, forward, images, mask, prompts, lengths, cfg, records):
    """One native forward, followed by replacement of only its final Normal logit."""
    if model.training:
        raise ValueError("V16 requires model.eval() and frozen inference")
    if not prompts or str(prompts[0]).strip().casefold() != "normal":
        raise ValueError("Expected Normal at class index 0 in native prompt order")
    result = forward(images, mask, prompts, lengths)
    visual, text, temperature = model._v16_capture
    del model._v16_capture
    ori, c, a = result
    # This is also a smoke test of capture placement, dtype and native logit scale.
    reconstructed = visual @ text / temperature
    if not torch.equal(a, reconstructed):
        raise ValueError("Captured embeddings do not reproduce native A logits exactly")
    _, info, normal = adapt_normal(visual, text, c, a, lengths, cfg)
    info["temperature"] = temperature
    if normal is None:
        changed = a.detach()
    else:
        # Use the native matmul shape/dtype; all anomaly columns remain bitwise unchanged.
        modified_text = text.detach().clone()
        modified_text[:, :, 0] = normal.to(text.dtype)
        candidate = visual @ modified_text / temperature
        changed = a.detach().clone()
        sizes = torch.as_tensor(lengths, device=a.device).reshape(-1)
        valid = torch.arange(a.shape[1], device=a.device)[None, :] < sizes[:, None]
        changed[..., 0][valid] = candidate[..., 0][valid]
        info["mean_abs_normal_logit_delta"] = float(
            (changed[..., 0][valid] - a[..., 0][valid]).abs().mean())
    if not torch.equal(changed[..., 1:], a[..., 1:]) or not changed.isfinite().all():
        raise ValueError("V16 changed anomaly logits or produced nonfinite values")
    records.append(info)
    return ori.detach(), c.detach(), changed
