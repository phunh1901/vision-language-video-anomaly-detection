"""Native VadCLIP control: original forward and source-sliced original objective.

Only the clean, pinned upstream Python source verified by v21_data.checkout is
accepted at runtime. No upstream training loop, test evaluator or argument parser
is executed to obtain the loss. The extracted objective is saved for review.
"""

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v21_model import FullPipeline, finite

NATIVE_SCHEMA = "v21_native_cosine_control_v1"


def _forward_call(node):
    return (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "model")


def _native_bindings(statements, end, functions):
    """Bind inputs by the actual forward/loss call sites, never by guessed names.

    UCF uses feat_lengths/text_labels/CLASM in the pinned source. The labels
    supplied by our loader are already encoded, so the source variable receiving
    them must be identified at the loss call, not taken from the raw loader label.
    """
    starts = [i for i, node in enumerate(statements[:end]) if _forward_call(node)]
    if not starts:
        raise ValueError("Pinned native model forward call missing before loss")
    start = starts[-1]
    node = statements[start]
    if (len(node.targets) != 1 or not isinstance(node.targets[0], (ast.Tuple, ast.List))
            or len(node.targets[0].elts) != 3
            or any(not isinstance(n, ast.Name) for n in node.targets[0].elts)
            or len(node.value.args) != 4 or not isinstance(node.value.args[-1], ast.Name)):
        raise ValueError("Unsupported pinned native forward input/output signature")
    text, c, a = [n.id for n in node.targets[0].elts]
    length_name = node.value.args[-1].id
    bindings = {text: "_native_text", c: "_native_c", a: "_native_a",
                length_name: "_native_lengths"}
    observed, labels, devices = set(), set(), set()
    for statement in statements[start + 1:end + 1]:
        for call in ast.walk(statement):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id in functions and call.args
                    and isinstance(call.args[0], ast.Name) and call.args[0].id in {c, a}):
                continue
            if (len(call.args) != 4 or any(not isinstance(n, ast.Name) for n in call.args)
                    or call.args[2].id != length_name):
                raise ValueError("Unsupported pinned native loss input signature")
            observed.add(call.args[0].id)
            labels.add(call.args[1].id)
            devices.add(call.args[3].id)
    if observed != {c, a} or len(labels) != 1 or len(devices) != 1:
        raise ValueError("Cannot unambiguously bind encoded labels/lengths for both native branches")
    label_name, device_name = next(iter(labels)), next(iter(devices))
    if label_name in bindings or device_name in bindings or label_name == device_name:
        raise ValueError("Conflicting native input aliases")
    bindings.update({label_name: "_native_targets", device_name: "_native_device"})
    return start, bindings


def native_objective(source, filename="pinned_train.py"):
    """Extract the loss backward slice from the pinned trainer, failing on unknown inputs.

    Keeps the actual native loss helpers and text contrastive expression, including
    its actual coefficient/reduction. In particular, no guessed paper loss is
    substituted. Expected tensors are logits [B,T,1]/[B,T,M], text [M,D], video
    labels [B,M], and valid lengths [B]. Return values are microbatch means.
    """
    tree = ast.parse(source, filename=filename)
    train = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"]
    if len(train) != 1:
        raise ValueError("Native objective requires one pinned train() function")
    candidates = []
    for parent in ast.walk(train[0]):
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for index, node in enumerate(value):
                    if isinstance(node, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == "loss" for t in node.targets):
                        candidates.append((value, index))
    if len(candidates) != 1:
        raise ValueError("Native objective requires one unambiguous loss assignment")
    statements, end = candidates[0]
    functions = {n.name: n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name != "train"}
    start, bindings = _native_bindings(statements, end, functions)
    inputs = set(bindings)
    globals_allowed = {"torch", "nn", "F", "np", "range", "len", "int", "float",
                       "enumerate", "zip", "min", "max", "sum", "abs"}
    needed, selected = {"loss", "loss1", "loss2", "loss3"}, []
    for node in reversed(statements[start + 1:end + 1]):
        stores = {n.id for n in ast.walk(node)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if not (stores & needed):
            continue
        if not isinstance(node, (ast.Assign, ast.AugAssign, ast.For)):
            raise ValueError("Unexpected native objective control flow")
        loads = {n.id for n in ast.walk(node)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        # A loop accumulator reads its old value, although AugAssign marks Store.
        accumulators = {n.target.id for n in ast.walk(node)
                        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)}
        internal = stores - accumulators if isinstance(node, ast.For) else set()
        needed = (needed - stores) | (loads - internal) | accumulators
        selected.append(node)
    helper_names = needed & set(functions)
    if not helper_names:
        raise ValueError("Pinned native loss helpers missing")
    pending = list(helper_names)
    while pending:
        name = pending.pop()
        dependencies = {n.id for n in ast.walk(functions[name])
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} & set(functions)
        new = dependencies - helper_names
        helper_names.update(new)
        pending.extend(sorted(new))
    unresolved = needed - inputs - globals_allowed - helper_names
    if unresolved:
        raise ValueError(f"Unresolved pinned native loss inputs: {sorted(unresolved)}")
    definitions = [n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name in helper_names]
    wrapper = ast.parse(
        "def extracted_loss(_native_text, _native_c, _native_a, _native_targets, _native_lengths, _native_device):\n"
        "    pass\n").body[0]
    aliases = [ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                          value=ast.Name(id=parameter, ctx=ast.Load()))
               for name, parameter in bindings.items()]
    wrapper.body = aliases + list(reversed(selected)) + [ast.parse(
        "return loss, loss1, loss2, loss3").body[0]]
    module = ast.fix_missing_locations(ast.Module(body=definitions + [wrapper], type_ignores=[]))
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np}
    exec(compile(module, filename, "exec"), namespace)
    extracted = ast.unparse(module)
    receipt = {"source_file": Path(filename).name,
               "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "extracted_sha256": hashlib.sha256(extracted.encode()).hexdigest(),
               "extracted_source": extracted,
               "extraction_contract": 2,
               "input_bindings": bindings,
               "native_loss_helpers": sorted(helper_names),
               "scope": "native objective; matched V21 loader/optimizer/budget, not original recipe"}

    def objective(c, a, targets, lengths, aux, space, config):
        if space != "native_cosine":
            raise ValueError("Native loss used with a different model family")
        if targets.shape != (a.shape[0], a.shape[-1]) or (
                (targets.sum(-1) <= 0).any() or
                ((targets[:, 0] > 0) & (targets[:, 1:].sum(-1) > 0)).any()):
            raise ValueError("Unknown/conflicting native video labels")
        sizes = torch.as_tensor(lengths, device=c.device)
        if bool(((sizes < 1) | (sizes > c.shape[1])).any()):
            raise ValueError("Invalid native valid sequence lengths")
        values = namespace["extracted_loss"](
            aux["encoder_text"], c, a, targets.float(), lengths, c.device)
        values = [v if isinstance(v, torch.Tensor) else c.new_tensor(v) for v in values]
        if any(v.numel() != 1 for v in values):
            raise ValueError("Native loss must return scalar microbatch means")
        for value in values:
            finite("native objective", value)
        return values[0].reshape(1), dict(zip(
            ("bce", "alignment", "contrastive"), [v.reshape(1) for v in values[1:]], strict=True))

    return objective, receipt


class NativePipeline(FullPipeline):
    """Keep native heads/fusion intact; call the captured unbound upstream forward.

    Capturing forward explicitly avoids recursion when evaluators instrument or
    replace CLIPVAD.forward. Diagnostics attach temporary hooks only during train.
    """

    def __init__(self, base, native_forward=None):
        nn.Module.__init__(self)
        if any(not hasattr(base, n) for n in ("classifier", "mlp1", "mlp2", "linear")):
            raise ValueError("Native control requires unmodified VadCLIP heads")
        self.base, self.space = base, "native_cosine"
        self._native_forward = native_forward or type(base).forward
        self.base.float().requires_grad_(True)
        self.base.clipmodel.requires_grad_(False).eval()

    def forward(self, images, padding_mask, text, lengths):
        with torch.autocast(device_type=images.device.type, enabled=False):
            result = self._native_forward(self.base, images.float(), padding_mask, text, lengths)
        if len(result) != 3:
            raise ValueError("Pinned native forward must return text, C logits, A logits")
        if self.training:
            for value in result:
                finite("native forward", value)
        return result

    def forward_with_aux(self, images, padding_mask, text, lengths):
        captured = []
        handle = self.base.linear.register_forward_hook(lambda module, args, value: captured.append(value))
        try:
            raw, c, a = self(images, padding_mask, text, lengths)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Native forward must execute the temporal output FC exactly once")
        sizes = torch.as_tensor(lengths, device=images.device)
        valid = torch.arange(images.shape[1], device=images.device)[None] < sizes[:, None]
        return raw, c, a, {"adapter": captured[0], "encoder_text": raw, "valid": valid}

    def parameter_groups(self):
        groups = dict.fromkeys(("frozen_clip", "temporal_adapter", "prompt_fusion",
                                "classification_head", "other_downstream"), 0)
        for name, parameter in self.named_parameters():
            groups[self.parameter_group(name)] += parameter.numel()
        return groups

    @staticmethod
    def parameter_group(name):
        if name.startswith(("base.classifier.", "base.mlp2.")):
            return "classification_head"
        return FullPipeline.parameter_group(name)

    def model_receipt(self):
        return {"space": self.space, "heads": "pinned native classifier/mlp2 and cosine alignment",
                "fusion": "pinned native forward", "encoder_precision": "float32",
                "poincare_mapping": False, "norm_conditioning": False,
                "scope": "architecture/objective control under matched V21 training recipe"}


def native_health(model, c, a, aux, parts, targets, lengths, config):
    """Valid TRAIN tokens only; raw gradients and score distributions, no frame GT."""
    gradients = {}
    for key, value in parts.items():
        values = torch.autograd.grad(value.mean(), (aux["adapter"], aux["encoder_text"]),
                                     retain_graph=True, allow_unused=True)
        for gradient in values:
            if gradient is not None:
                finite(f"native {key} diagnostic gradient", gradient)
        gradients[key] = {side: {"connected": g is not None,
                                "norm": 0. if g is None else float(g.detach().double().norm())}
                          for side, g in zip(("visual", "text"), values, strict=True)}
    temporal = []
    c_gradient = torch.autograd.grad(parts["bce"].mean(), c, retain_graph=True, allow_unused=True)[0]
    if c_gradient is not None:
        finite("native BCE gradient at raw C", c_gradient)
    for i, length in enumerate(lengths):
        n = int(length)
        z, align = c[i, :n, 0].detach().double(), a[i, :n].detach().double()
        odds = torch.logsumexp(align[:, 1:], -1) - align[:, 0]
        fp32_probability = c[i, :n, 0].detach().float().sigmoid()
        k = 1 + n // 16
        pooled = fp32_probability.topk(k).values.mean()
        temporal.append({"valid_steps": n, "abnormal_video_label": bool(targets[i, 0] == 0),
                         "classification_logit_min": float(z.min()),
                         "classification_logit_max": float(z.max()),
                         "fp32_exact_zero_count": int((fp32_probability == 0).sum()),
                         "fp32_exact_one_count": int((fp32_probability == 1).sum()),
                         "native_topk": k, "native_pooled_probability_fp32": float(pooled),
                         "wrong_saturated_pool": bool(pooled == (0 if targets[i, 0] == 0 else 1)),
                         "bce_gradient_at_raw_c_norm": (float(c_gradient[i, :n].detach().double().norm())
                                                        if c_gradient is not None else None),
                         "classification_probability_mean": float(z.sigmoid().mean()),
                         "classification_probability_std": float(z.sigmoid().std(correction=0)),
                         "alignment_log_odds_mean": float(odds.mean()),
                         "alignment_log_odds_std": float(odds.std(correction=0))})
    return {"scope": "TRAIN diagnostic; raw native loss gradients, not weighted objective gradients",
            "microbatch_size": len(lengths), "loss_gradients": gradients, "temporal": temporal,
            "adapter_norm_median": float(aux["adapter"].detach()[aux["valid"]].norm(dim=-1).median()),
            "model": model.model_receipt()}


def native_preflight(model, pack, ids, labels, abnormal, train, args, prompts,
                     make_mask, device, objective, loss_config, video_input, ignore_padding_mask=False):
    selected = np.r_[train[~abnormal[train]][:2], train[abnormal[train]][:2]]
    if not len(train[~abnormal[train]]) or not len(train[abnormal[train]]):
        raise ValueError("Native preflight requires normal and abnormal TRAIN videos")
    probes = []
    model.eval()
    for index in selected:
        model.zero_grad(set_to_none=True)
        x, lengths, _ = video_input(pack.array(pack.features[ids[int(index)]]),
                                   args.visual_length, device, np.random.default_rng(42))
        mask = None if ignore_padding_mask else make_mask(lengths, args.visual_length).to(device)
        _, c, a, aux = model.forward_with_aux(x, mask, prompts, lengths)
        loss, parts = objective(c, a, labels[int(index)].to(device)[None], lengths,
                                aux, "native_cosine", loss_config)
        probes.append({"video_id": ids[int(index)], **native_health(
            model, c, a, aux, parts, labels[int(index)].to(device)[None], lengths, loss_config)})
        loss.mean().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                finite(f"native preflight gradient {name}", p.grad)
        receipt = model.gradient_receipt()
        if not all(receipt[k] for k in ("classification_head", "temporal_adapter", "prompt_tokens")):
            raise RuntimeError(f"Disconnected native preflight gradient: {receipt}")
    model.zero_grad(set_to_none=True)
    return {"test_data_used": False, "optimizer_updates": 0, "calibrated": False,
            "conditioning": False, "classifier_bias_calibrated": False, "learning_health": probes}


def load_native_model(base, path, expected_sha256, initial_v0_sha256, device, native_forward=None):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Native checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != NATIVE_SCHEMA or payload.get("space") != "native_cosine"
            or payload.get("initial_v0_sha256") != initial_v0_sha256):
        raise ValueError("Native checkpoint schema/provenance mismatch")
    model = NativePipeline(base, native_forward).to(device)
    actual_forward = inspect.getsource(model._native_forward)
    if hashlib.sha256(actual_forward.encode()).hexdigest() != payload["native_forward_sha256"]:
        raise ValueError("Native train/eval forward source mismatch")
    model.load_downstream(payload["downstream"])
    model.eval()
    receipt = {"schema": NATIVE_SCHEMA, "trained_checkpoint_sha256": expected_sha256,
               "initial_v0_sha256": initial_v0_sha256, "epoch": payload["epoch"],
               "native_forward_sha256": payload["native_forward_sha256"],
               "native_objective": payload["native_objective"],
               "parameter_groups": model.parameter_groups(), "model": model.model_receipt()}
    return model, receipt
