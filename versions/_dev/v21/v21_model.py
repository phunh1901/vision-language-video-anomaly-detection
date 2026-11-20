"""Full downstream V21: temporal/prompt training, CLS prototype and tangent fusion.

CLIP image/text parameters stay frozen, but the text encoder's input graph stays
enabled so gradients reach learned prompt tokens. Uses pinned VadCLIP's adapter
and prompt encoder methods; never its old Euclidean classification/alignment.
"""

import hashlib
import inspect
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from v21_geometry import cone_energy, expmap0, logmap0, poincare_distance

SCHEMA = "v21_full_downstream_v5_sanity"


def finite(name, tensor):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Non-finite tensor before loss: {name}")


def pooled_binary_loss(logits, target, k):
    """BCE(mean(top-k sigmoid(z)), y), exactly in log space, without clamping p."""
    finite("classification logits", logits)
    z = logits.float().topk(k).values
    log_p = torch.logsumexp(F.logsigmoid(z), dim=0) - math.log(k)
    log_not_p = torch.logsumexp(F.logsigmoid(-z), dim=0) - math.log(k)
    return -target * log_p - (1 - target) * log_not_p


class GeometryHead(nn.Module):
    def __init__(self, dimension=512, curvature=.5, scale=.7, temperature=.07,
                 chunk_size=64, space="poincare", max_tangent_norm=None,
                 curvature_min=.1, curvature_max=1., text_projection=True,
                 geometry_dtype="float32", encoder_precision="inherit",
                 curvature_trainable=True, max_relative_radius=None):
        super().__init__()
        if (space not in {"cosine", "poincare"}
                or not 0 < curvature_min < curvature < curvature_max
                or not all(math.isfinite(v) for v in (curvature_min, curvature, curvature_max))):
            raise ValueError("Invalid model geometry")
        if dimension < 1 or scale <= 0 or temperature <= 0 or chunk_size < 1:
            raise ValueError("Invalid geometry dimensions/scales")
        self.space, self.temperature, self.chunk_size = space, temperature, chunk_size
        if max_tangent_norm is not None and (
                not math.isfinite(max_tangent_norm) or max_tangent_norm <= 0):
            raise ValueError("max_tangent_norm must be finite and positive")
        self.max_tangent_norm = max_tangent_norm
        if max_relative_radius is not None and (
                not math.isfinite(max_relative_radius) or not 0 < max_relative_radius < 1):
            raise ValueError("max_relative_radius must be in (0,1)")
        if max_relative_radius is not None and max_tangent_norm is not None:
            raise ValueError("Select smooth radius limiting or legacy hard clipping, not both")
        self.max_relative_radius = max_relative_radius
        self.register_buffer("visual_norm_scale", torch.ones(()))
        self.register_buffer("text_norm_scale", torch.ones(()))
        if geometry_dtype not in {"float32", "float64"}:
            raise ValueError("geometry_dtype must be float32 or float64")
        self.geometry_dtype = geometry_dtype
        if encoder_precision not in {"inherit", "float32"}:
            raise ValueError("encoder_precision must be inherit or float32")
        self.encoder_precision = encoder_precision
        self.visual = nn.Linear(dimension, dimension, bias=False)
        self.text = nn.Linear(dimension, dimension, bias=False) if text_projection else nn.Identity()
        with torch.no_grad():
            self.visual.weight.copy_(torch.eye(dimension) / math.sqrt(dimension))
            if text_projection:
                self.text.weight.copy_(torch.eye(dimension) / math.sqrt(dimension))
        self.curvature_min, self.curvature_max = curvature_min, curvature_max
        self.raw_c = nn.Parameter(torch.tensor(math.log(
            (curvature - curvature_min) / (curvature_max - curvature))))
        self.raw_v = nn.Parameter(torch.tensor(math.log(math.expm1(scale))))
        self.raw_t = nn.Parameter(torch.tensor(math.log(math.expm1(scale))))
        self.prototype = nn.Parameter(torch.randn(dimension) * .01)
        self.bias = nn.Parameter(torch.zeros(()))
        if space == "cosine" or not curvature_trainable:
            self.raw_c.requires_grad_(False)

    @property
    def curvature(self):
        return self.curvature_min + (self.curvature_max - self.curvature_min) * self.raw_c.sigmoid()

    def map_visual(self, features):
        return self.map_tangent(self.visual_tangent(features))

    def map_text(self, features):
        return self.map_tangent(self.text_tangent(features))

    def visual_tangent(self, features):
        return F.softplus(self.raw_v) * self.visual(features.float()) / self.visual_norm_scale

    def text_tangent(self, features):
        return F.softplus(self.raw_t) * self.text(features.float()) / self.text_norm_scale

    @torch.no_grad()
    def set_modality_norms(self, visual, text):
        if not all(math.isfinite(n) and n > 1e-8 for n in (visual, text)):
            raise ValueError("Conditioning norms must be finite and positive")
        self.visual_norm_scale.fill_(visual)
        self.text_norm_scale.fill_(text)

    def map_tangent(self, tangent):
        if self.space != "poincare":
            return tangent
        tangent = tangent.double() if self.geometry_dtype == "float64" else tangent.float()
        if self.max_relative_radius is not None:
            # Smooth radial compression; direction and relative norm information
            # remain variable. This is a disclosed stabilization, not paper Eq1 verbatim.
            limit = math.atanh(self.max_relative_radius) / self.curvature.to(tangent).sqrt()
            z = tangent.norm(dim=-1, keepdim=True) / limit
            factor = torch.where(z > 1e-7, z.tanh() / z.clamp_min(1e-7), 1 - z.square() / 3)
            tangent = tangent * factor
        if self.max_tangent_norm is not None:
            norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            tangent = tangent * (self.max_tangent_norm / norm).clamp(max=1.)
        return expmap0(tangent, self.curvature)

    def classify(self, visual):
        if self.space == "poincare":
            prototype = self.map_tangent(self.prototype)
            distance = poincare_distance(visual, prototype, self.curvature)
        else:
            distance = torch.linalg.vector_norm(visual - self.prototype, dim=-1)
        return (-distance + self.bias).unsqueeze(-1)

    def align(self, visual, text):
        if self.space == "cosine":
            return F.normalize(visual, dim=-1) @ F.normalize(text, dim=-1).transpose(-1, -2) / self.temperature
        return torch.cat([
            -poincare_distance(visual[:, i:i + self.chunk_size, None], text[:, None],
                               self.curvature) / self.temperature
            for i in range(0, visual.shape[1], self.chunk_size)
        ], dim=1)

    def receipt(self):
        return {"space": self.space, "curvature_magnitude": float(self.curvature.detach()),
                "lambda_visual": float(F.softplus(self.raw_v).detach()),
                "lambda_text": float(F.softplus(self.raw_t).detach()),
                "temperature": self.temperature, "chunk_size": self.chunk_size,
                "max_tangent_norm": self.max_tangent_norm,
                "text_projection": isinstance(self.text, nn.Linear),
                "geometry_dtype": self.geometry_dtype,
                "encoder_precision": self.encoder_precision,
                "boundary_epsilon": 1e-12 if self.geometry_dtype == "float64" else 1e-5,
                "curvature_trainable": self.raw_c.requires_grad,
                "max_relative_radius": self.max_relative_radius,
                "visual_norm_scale": float(self.visual_norm_scale),
                "text_norm_scale": float(self.text_norm_scale),
                "parameters": sum(p.numel() for p in self.parameters()),
                "scope": "full CLS, tangent-space prompt fusion and alignment"}


class FullPipeline(nn.Module):
    def __init__(self, base, space, geometry_config):
        super().__init__()
        required = ("encode_video", "encode_textprompt", "mlp1", "clipmodel")
        if any(not hasattr(base, name) for name in required):
            raise ValueError("Pinned VadCLIP lacks the expected adapter/prompt/CLIP API")
        self.base = base
        self.geometry_config = dict(geometry_config)
        self.space = space
        # Old Euclidean C block is replaced, not left trainable but unused.
        self.base.classifier = nn.Identity()
        self.base.mlp2 = nn.Identity()
        for name, module in self.base.named_children():
            if name != "clipmodel":
                module.float()
        self.base.requires_grad_(True)
        self.base.clipmodel.requires_grad_(False).eval()
        self.geometry = GeometryHead(space=space, **geometry_config)
        if self.geometry.encoder_precision == "float32":
            # CLIP's loader can explicitly store half weights and cast its inputs
            # to self.dtype. Disabling autocast alone does not undo those casts.
            self.base.float()

    def train(self, mode=True):
        super().train(mode)
        self.base.clipmodel.eval()
        return self

    def forward_with_aux(self, images, padding_mask, text, lengths):
        if self.geometry.encoder_precision == "float32":
            with torch.autocast(device_type=images.device.type, enabled=False):
                visual = self.base.encode_video(images.float(), padding_mask, lengths)
                # Frozen weights, enabled input graph: prompts must still learn.
                text_raw = self.base.encode_textprompt(text)
        else:
            visual = self.base.encode_video(images, padding_mask, lengths)
            text_raw = self.base.encode_textprompt(text)
        if self.training:
            finite("adapter output", visual)
            finite("text encoder output", text_raw)
        if visual.ndim != 3 or text_raw.ndim != 2:
            raise ValueError("Expected adapter [B,T,d] and prompt [M,d]")
        if visual.shape[-1] != self.geometry.visual.in_features or text_raw.shape[-1] != visual.shape[-1]:
            raise ValueError("Embedding dimension mismatch")
        lengths = torch.as_tensor(lengths, device=images.device)
        valid = torch.arange(images.shape[1], device=images.device)[None] < lengths[:, None]
        with torch.autocast(device_type=images.device.type, enabled=False):
            visual_tangent = self.geometry.visual_tangent(visual)
            text_tangent = self.geometry.text_tangent(text_raw)
            x = self.geometry.map_tangent(visual_tangent)
            t = self.geometry.map_tangent(text_tangent)
            logits_c = self.geometry.classify(x)
            if self.space == "poincare":
                tangent_x = logmap0(x, self.geometry.curvature)
                tangent_t = logmap0(t, self.geometry.curvature)
            else:
                tangent_x, tangent_t = x, t
            attention = logits_c.sigmoid() * valid.unsqueeze(-1)
            prompt = F.normalize((attention * tangent_x).sum(1), dim=-1, eps=1e-6)
            fused = self.base.mlp1((prompt[:, None] + tangent_t[None]).float()) + tangent_t[None]
            instance_text = self.geometry.map_tangent(fused)
            logits_a = self.geometry.align(x, instance_text)
        if self.training:
            finite("visual ball", x)
            finite("instance text ball", instance_text)
            finite("alignment logits", logits_a)
        aux = {"visual": x, "text": instance_text, "valid": valid,
               "adapter": visual, "encoder_text": text_raw, "class_text": t,
               "visual_tangent": visual_tangent, "text_tangent": text_tangent,
               "fused_tangent": fused,
               "curvature": self.geometry.curvature}
        return text_raw, logits_c, logits_a, aux

    def forward(self, images, padding_mask, text, lengths):
        raw, c, a, _ = self.forward_with_aux(images, padding_mask, text, lengths)
        return raw, c, a

    def parameter_groups(self):
        groups = {"frozen_clip": 0, "temporal_adapter": 0, "prompt_fusion": 0,
                  "geometry_classifier": 0, "other_downstream": 0}
        for name, parameter in self.named_parameters():
            group = self.parameter_group(name)
            if group == "frozen_clip":
                if parameter.requires_grad:
                    raise ValueError("CLIP parameter unexpectedly trainable")
            groups[group] += parameter.numel()
        return groups

    @staticmethod
    def parameter_group(name):
        if name.startswith("base.clipmodel."):
            return "frozen_clip"
        if name.startswith("geometry."):
            return "geometry_classifier"
        if name.startswith(("base.text_prompt", "base.mlp1")):
            return "prompt_fusion"
        if name.startswith(("base.temporal", "base.gc", "base.linear", "base.disAdj",
                            "base.frame_position")):
            return "temporal_adapter"
        return "other_downstream"

    def gradient_receipt(self):
        result = {group: False for group in self.parameter_groups()}
        for name, p in self.named_parameters():
            if p.grad is not None and bool(p.grad.abs().sum() > 0):
                result[self.parameter_group(name)] = True
        result["prompt_tokens"] = any(
            p.grad is not None and bool(p.grad.abs().sum() > 0)
            for name, p in self.named_parameters() if name.startswith("base.text_prompt"))
        result["clip_parameter_gradients"] = result.pop("frozen_clip")
        return result

    def downstream_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if not name.startswith("base.clipmodel.")}

    def load_downstream(self, state):
        expected = {name for name in self.state_dict() if not name.startswith("base.clipmodel.")}
        if set(state) != expected:
            raise ValueError("Full downstream checkpoint schema/keys mismatch")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith("base.clipmodel.") for k in result.missing_keys):
            raise ValueError("Unexpected missing downstream model weights")


def topk_size(count, abnormal):
    return min(count, 1 + count // 16) if abnormal else 1


def full_loss(logits_c, logits_a, targets, lengths, aux, space, config):
    """Eq.(13) per video. Only video labels; temporal padding is excluded."""
    if targets.ndim != 2 or targets.shape != (logits_a.shape[0], logits_a.shape[-1]):
        raise ValueError("Video labels and logits have different class/batch axes")
    if (targets.sum(-1) <= 0).any() or ((targets[:, 0] > 0) & (targets[:, 1:].sum(-1) > 0)).any():
        raise ValueError("Unknown/conflicting video labels")
    components = {key: [] for key in ("bce", "alignment", "entailment", "contrastive")}
    for batch in range(len(targets)):
        count = int(lengths[batch])
        if count < 1 or count > logits_a.shape[1]:
            raise ValueError("Invalid valid sequence length")
        target = targets[batch].float()
        abnormal = bool(target[0] == 0)
        k = topk_size(count, abnormal)
        c = logits_c[batch, :count, 0].float()
        a = logits_a[batch, :count].float()
        finite("alignment loss input", a)
        components["bce"].append(pooled_binary_loss(c, 1 - target[0], k))
        pooled = a.topk(k, dim=0).values.mean(0)
        alignment_temperature = config.get("alignment_loss_temperature", 1.)
        if not math.isfinite(alignment_temperature) or alignment_temperature <= 0:
            raise ValueError("alignment_loss_temperature must be finite and positive")
        components["alignment"].append(-(target / target.sum() * F.log_softmax(
            pooled / alignment_temperature, -1)).sum())
        x, text = aux["visual"][batch, :count], aux["text"][batch]
        pairs = []
        for category in torch.nonzero(target > 0, as_tuple=False).flatten():
            positions = a[:, category].topk(k).indices
            pairs.append(cone_energy(x[positions], text[category], aux["curvature"], space,
                                     config["cone_aperture_scale"], config["entail_regularizer"]).mean())
        components["entailment"].append(torch.stack(pairs).mean())
        if space == "poincare":
            distance = poincare_distance(text[0], text[1:], aux["curvature"])
        else:
            distance = torch.linalg.vector_norm(text[0] - text[1:], dim=-1)
        components["contrastive"].append((config["separation_margin"] - distance).clamp_min(0).sum())
    values = {key: torch.stack(value) for key, value in components.items()}
    total = (values["bce"] + values["alignment"] + config["psi"] * values["entailment"]
             + config["phi"] * values["contrastive"])
    return total, values


def load_full_model(base, space, geometry_config, checkpoint, expected_sha256,
                    initial_v0_sha256, device):
    path = Path(checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Trained downstream checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA or payload.get("space") != space
            or payload.get("initial_v0_sha256") != initial_v0_sha256
            or payload.get("geometry_config") != geometry_config):
        raise ValueError("Full V21 checkpoint recipe/warm-start mismatch")
    if payload.get("train_config", {}).get("single_temperature") and (
            payload.get("loss_config", {}).get("alignment_loss_temperature", 1.) != 1.):
        raise ValueError("Checkpoint violates the single-temperature training/inference contract")
    model = FullPipeline(base, space, geometry_config).to(device)
    model.load_downstream(payload["downstream"])
    model.eval()
    provenance = {
        "schema": SCHEMA, "trained_checkpoint_sha256": expected_sha256,
        "initial_v0_sha256": initial_v0_sha256, "epoch": payload["epoch"],
        "parameter_groups": model.parameter_groups(), "trainable_scope": "all downstream; CLIP frozen",
        "encoder_api_source_sha256": {
            name: hashlib.sha256(inspect.getsource(getattr(type(base), name)).encode()).hexdigest()
            for name in ("encode_video", "encode_textprompt")
        },
    }
    return model, provenance
