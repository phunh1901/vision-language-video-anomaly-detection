"""Train-only conditioning selection, optimizer groups and per-loss diagnostics."""

import math

import numpy as np
import torch
from v21_geometry import cone_energy


def probe_indices(train, labels, abnormal, limit=32, seed=42):
    """Deterministic normal/abnormal sample covering available positive classes."""
    if type(limit) is not int or limit < 2:
        raise ValueError("initialization_probe_videos must be an integer >=2")
    rng = np.random.default_rng(seed)
    train = np.asarray(train, dtype=int)
    normal = rng.permutation(train[~abnormal[train]]).tolist()
    positive = rng.permutation(train[abnormal[train]]).tolist()
    if not normal or not positive:
        raise ValueError("Conditioning requires both normal and abnormal TRAIN videos")
    cover = []
    for category in range(1, labels.shape[1]):
        candidates = [i for i in positive if bool(labels[i, category] > 0)]
        if candidates and not any(bool(labels[i, category] > 0) for i in cover):
            cover.append(candidates[0])
    if len(cover) >= limit:
        raise ValueError("Probe budget cannot cover positive classes and normal; increase it")
    normal_budget = min(len(normal), max(1, min(limit // 2, limit - len(cover))))
    pos_budget = min(len(positive), limit - normal_budget)
    positives = cover + [i for i in positive if i not in cover]
    selected = normal[:normal_budget] + positives[:pos_budget]
    return np.asarray(selected, dtype=int)


def joint_optimizer(model, config):
    """Retain Adam; explicitly exclude frozen params and optional scalar/bias/norm decay."""
    buckets = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            no_decay = config.get("no_decay_scalars_bias_norm", False) and p.ndim < 2
            decay = 0. if no_decay else config["weight_decay"]
            buckets.setdefault(decay, []).append(p)
    kind = config.get("optimizer", "Adam")
    if kind not in {"Adam", "AdamW"}:
        raise ValueError(f"Unknown optimizer: {kind}")
    return getattr(torch.optim, kind)(
        [{"params": ps, "weight_decay": decay} for decay, ps in buckets.items()],
        lr=config["learning_rate"], betas=tuple(config.get("optimizer_betas", (.9, .999))))


def learning_health(model, c, a, aux, parts, targets, lengths, loss_config):
    """Detached diagnostics on valid TRAIN steps. No frame GT or model selection.

    Raw and weighted gradient norms distinguish a disconnected loss from a small
    coefficient. Hinge satisfaction and low temporal variance are logged, not
    treated as automatic failures or used as new training objectives.
    """
    gradients = {}
    factors = {"bce": 1., "alignment": 1., "entailment": loss_config["psi"],
               "contrastive": loss_config["phi"]}
    inputs = (aux["adapter"], aux["encoder_text"])
    for key in factors:
        values = torch.autograd.grad(parts[key].mean(), inputs, retain_graph=True,
                                     allow_unused=True)
        entry = {}
        for name, gradient in zip(("visual", "text"), values, strict=True):
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"Non-finite per-loss {key}/{name} gradient")
            norm = 0. if gradient is None else float(gradient.detach().norm(dtype=torch.float64))
            entry[name] = {"connected": gradient is not None, "norm": norm,
                           "weighted_norm": abs(factors[key]) * norm}
        gradients[key] = entry
    temporal, fractions, pair_count = [], {}, 0
    for i, length in enumerate(lengths):
        n = int(length)
        z, align = c[i, :n, 0].detach().double(), a[i, :n].detach().double()
        odds = torch.logsumexp(align[:, 1:], -1) - align[:, 0]
        temporal.append({"valid_steps": n, "abnormal_video_label": bool(targets[i, 0] == 0),
                         "classification_logit_std": float(z.std(correction=0)),
                         "classification_probability_std": float(z.sigmoid().std(correction=0)),
                         "alignment_log_odds_std": float(odds.std(correction=0))})
        k = min(n, 1 + n // 16) if targets[i, 0] == 0 else 1
        for category in torch.nonzero(targets[i] > 0, as_tuple=False).flatten():
            # full_loss mines pairs after its FP32 loss cast; match that cast
            # rather than selecting different pairs at near-ties in FP64.
            positions = align.float()[:, category].topk(k).indices
            _, stats = cone_energy(aux["visual"][i, positions].detach(),
                                   aux["text"][i, category].detach(), aux["curvature"].detach(),
                                   aperture_scale=loss_config["cone_aperture_scale"],
                                   regularizer=loss_config["entail_regularizer"], diagnostics=True)
            for key, value in stats.items():
                if key != "angle_method":
                    fractions[key] = fractions.get(key, 0.) + float(value) * k
            pair_count += k
    radii, compression = {}, {}
    for key, values in (("visual", aux["visual"][aux["valid"]]),
                        ("class_text", aux["class_text"].reshape(-1, aux["class_text"].shape[-1])),
                        ("fused_text", aux["text"].reshape(-1, aux["text"].shape[-1]))):
        radius = values.detach().double().norm(dim=-1) * aux["curvature"].detach().double().sqrt()
        radii[key] = dict(zip(("min", "median", "p95", "max"),
                             radius.quantile(torch.tensor([0., .5, .95, 1.], device=radius.device,
                                                          dtype=torch.float64)).tolist(), strict=True))
        cap = model.geometry.max_relative_radius
        if cap is not None and radii[key]["max"] > cap + 1e-6:
            raise RuntimeError(f"Configured relative-radius bound violated: {key}")
    if not all(math.isfinite(v) for row in radii.values() for v in row.values()):
        raise RuntimeError("Non-finite geometry health statistics")
    if model.geometry.max_relative_radius is not None:
        limit = math.atanh(model.geometry.max_relative_radius) / (
            aux["curvature"].detach().double().sqrt())
        for key, values in (("visual", aux["visual_tangent"][aux["valid"]]),
                            ("class_text", aux["text_tangent"]),
                            ("fused_text", aux["fused_tangent"])):
            ratio = values.detach().double().norm(dim=-1) / limit
            if not bool(torch.isfinite(ratio).all()):
                raise RuntimeError(f"Non-finite pre-compression tangent norms: {key}")
            compression[key] = {"norm_over_limit_fraction": float((ratio > 1).double().mean()),
                                "max_norm_over_limit": float(ratio.max())}
    return {"scope": "training only; no frame labels or threshold selection", "radii": radii,
            "gradient_scope": "mean loss of this microbatch; compare like batch sizes",
            "microbatch_size": len(lengths), "tangent_compression": compression,
            "cone": {"angle_method": "atan2", "pairs": pair_count,
                     **{key: value / pair_count for key, value in fractions.items()}},
            "loss_gradients": gradients, "temporal": temporal,
            "curvature_trainable": model.geometry.raw_c.requires_grad}
