"""Poincare geometry, curvature -c; fp32/fp64 math for the full V21 pipeline."""

import torch


def ball_epsilon(x, eps=None):
    """Boundary guard scaled to the precision actually used on the manifold."""
    return (1e-12 if x.dtype == torch.float64 else 1e-5) if eps is None else eps


def project_ball(x, c, eps=None):
    eps = ball_epsilon(x, eps)
    c = c.to(dtype=x.dtype, device=x.device)
    maximum = (1 - eps) / c.sqrt()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (maximum / norm.clamp_min(torch.finfo(x.dtype).tiny)).clamp(max=1.)


def expmap0(v, c, eps=None):
    """Equation (1): T_0 B_c^d -> B_c^d, with a continuous zero limit."""
    c = c.to(dtype=v.dtype, device=v.device)
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    z = c.sqrt() * norm
    # Avoid 0/0 in the unselected branch: autograd still evaluates both branches.
    scale = torch.where(z > 1e-7, z.tanh() / z.clamp_min(1e-7), 1 - z.square() / 3)
    return project_ball(v * scale, c, eps)


def logmap0(x, c, eps=None):
    eps = ball_epsilon(x, eps)
    c = c.to(dtype=x.dtype, device=x.device)
    x = project_ball(x, c, eps)
    z = c.sqrt() * torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    z = z.clamp(max=1 - eps)
    scale = torch.where(z > 1e-7, torch.atanh(z) / z.clamp_min(1e-7), 1 + z.square() / 3)
    return x * scale


def poincare_distance(x, y, c, eps=None):
    """Equation (3), evaluated using an equivalent stable asinh identity.

    x/y broadcast over all leading axes; the last axis is the coordinate axis.
    d = 2/sqrt(c) * asinh(sqrt(c)*||x-y|| / sqrt((1-c||x||²)(1-c||y||²))).
    Direct vector differences avoid cancellation at x==y; norm has zero gradient
    at the origin in PyTorch. Geometry is calculated in fp32/fp64, never fp16.
    """
    eps = ball_epsilon(x, eps)
    c = c.to(dtype=x.dtype, device=x.device)
    x, y = project_ball(x, c, eps), project_ball(y, c, eps)
    dx = (1 - c * x.square().sum(-1)).clamp_min(eps)
    dy = (1 - c * y.square().sum(-1)).clamp_min(eps)
    delta = torch.linalg.vector_norm(x - y, dim=-1)
    return 2 / c.sqrt() * torch.asinh(c.sqrt() * delta / (dx * dy).sqrt())


def mobius_add(x, y, c):
    c = c.to(dtype=x.dtype, device=x.device)
    xy = (x * y).sum(-1, keepdim=True)
    xx, yy = x.square().sum(-1, keepdim=True), y.square().sum(-1, keepdim=True)
    return ((1 + 2 * c * xy + c * yy) * x + (1 - c * xx) * y) / (
        1 + 2 * c * xy + c.square() * xx * yy).clamp_min(1e-8)


def cone_energy(child, parent, c, space="poincare", aperture_scale=.1, regularizer=.01,
                diagnostics=False):
    """Eq.(11), using text as cone anchor, visual instance as child.

    Paper does not specify cone aperture scale, lambda_reg, or pair mining.
    These choices are explicit config. Poincare aperture follows Ganea 2018
    after scaling coordinates by sqrt(c); angle uses conformality and the
    direction log_parent(child). Euclidean control uses its straight tangent.
    A finite aperture guard is used near the cone domain's excluded origin.
    """
    c = c.to(dtype=parent.dtype, device=parent.device)
    eps = 1e-6
    radius = torch.linalg.vector_norm(parent, dim=-1)
    if space == "poincare":
        direction = mobius_add(-parent, child, c)
        aperture_denominator = c.sqrt() * radius
        aperture_argument = aperture_scale * (1 - c * radius.square()) / (
            aperture_denominator.clamp_min(eps))
    else:
        direction = child - parent
        aperture_denominator = radius
        aperture_argument = aperture_scale / aperture_denominator.clamp_min(eps)
    direction_norm = torch.linalg.vector_norm(direction, dim=-1)
    # atan2 gives the same exterior angle without clipping valid cosines near -1.
    # Compute the perpendicular vector directly, not sqrt(1-cosine**2), which
    # would lose small angles through cancellation. Protect atan2(0,0) first.
    unit_parent = parent / radius.unsqueeze(-1).clamp_min(eps)
    parallel = (unit_parent * direction).sum(-1)
    perpendicular = torch.linalg.vector_norm(
        direction - parallel.unsqueeze(-1) * unit_parent, dim=-1)
    degenerate = (direction_norm < eps) | (radius < eps)
    angle = torch.atan2(perpendicular, torch.where(degenerate, torch.ones_like(parallel), parallel))
    angle = torch.where(degenerate, torch.zeros_like(angle), angle)
    aperture = torch.asin(aperture_argument.clamp(eps, 1 - eps))
    energy = (angle - aperture).clamp_min(0) - regularizer * angle
    if not diagnostics:
        return energy
    cosine = (parent * direction).sum(-1) / (radius * direction_norm).clamp_min(eps)
    return energy, {
        "angle_method": "atan2",
        "legacy_acos_would_clip_fraction": ((cosine < -1 + eps) | (cosine > 1 - eps)).double().mean().detach(),
        "asin_clip_fraction": ((aperture_argument < eps) | (aperture_argument > 1 - eps)).double().mean().detach(),
        "legacy_cosine_denominator_would_clip_fraction": ((radius * direction_norm) < eps).double().mean().detach(),
        "parent_norm_guard_fraction": (radius < eps).double().mean().detach(),
        "aperture_denominator_guard_fraction": (aperture_denominator < eps).double().mean().detach(),
        "degenerate_fraction": degenerate.double().mean().detach(),
        "hinge_active_fraction": (angle > aperture).double().mean().detach(),
    }

