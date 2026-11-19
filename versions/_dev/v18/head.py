"""Residual scoring from CLIP appearance and multi-scale temporal differences."""

import torch
from torch import nn
from torch.nn import functional as F

TEMPORAL_LAGS = (1, 2, 4)


def anomaly_logits(c, a):
    a = a.float()
    return c.float().squeeze(-1), a[..., 1:].logsumexp(-1) - a[..., 0]


class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = nn.Linear(512, 64)
        self.temporal = nn.ModuleList(
            nn.Linear(512, 128, bias=False) for _ in TEMPORAL_LAGS
        )
        self.output = nn.Sequential(nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, clip, use_temporal):
        # Use the same feature precision as the training cache.
        features = F.normalize(clip.to(torch.float16).float(), dim=-1) * (512 ** 0.5)
        visual = self.clip(features)
        if use_temporal:
            for lag, projection in zip(TEMPORAL_LAGS, self.temporal, strict=True):
                if len(features) <= lag:
                    continue
                # Linear projection commutes with subtraction. Each direction has
                # its own weights, and missing neighbors contribute zero.
                past, future = projection(features).chunk(2, dim=-1)
                past = F.pad(past[lag:] - past[:-lag], (0, 0, lag, 0))
                future = F.pad(future[:-lag] - future[lag:], (0, 0, 0, lag))
                visual = visual + (past + future) / len(TEMPORAL_LAGS)
        return 2 * self.output(visual).squeeze(-1).tanh()


def load_head(path, strength, use_temporal, device):
    head = TemporalHead().to(device)
    head.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return head.eval(), strength, use_temporal


@torch.no_grad()
def apply_head(bundle, clip, lengths, c, a, primary):
    if bundle is None:
        return c, a
    head, strength, use_temporal = bundle
    if strength == 0:
        return c, a
    steps = int(torch.as_tensor(lengths).sum())
    delta = head(clip.reshape(-1, 512)[:steps], use_temporal) * strength
    if not torch.isfinite(delta).all():
        raise ValueError("Non-finite V18 residual")
    delta = F.pad(delta, (0, clip.shape[0] * clip.shape[1] - steps)).reshape(clip.shape[:2])
    if primary == "C":
        return c + delta[..., None].to(c.dtype), a
    modified = a.clone()
    modified[..., 0] -= delta.to(a.dtype)
    return c, modified
