"""Train the entire V21 downstream model; freeze only pretrained CLIP weights.

Video-label supervision only. Development video AP selects the saved epoch;
official frame test annotations never enter this process. A separate native
cosine control keeps its original forward/objective under the matched recipe.
"""

import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import v21_data as data
from benchmark_reporting import render_table
from sklearn.metrics import average_precision_score
from v21_diagnostics import StageTrace, append_json, check_batch, summarize_training
from v21_flow import EncoderFlowAudit
from v21_health import joint_optimizer, learning_health, probe_indices
from v21_model import SCHEMA, FullPipeline, full_loss, topk_size
from v21_native import (
    NATIVE_SCHEMA,
    NativePipeline,
    native_health,
    native_objective,
    native_preflight,
)
from v21_optimizer import optimizer_update
from v21_recipe import FullCropPack, recipe_batches, repository_video_input
from v21_schedule import FromScratchTraining, TwoStageTraining


def save_checkpoint(payload, destination):
    """Publish a complete checkpoint atomically; keep the previous one on write failure."""
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".writing")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def downstream_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith("clipmodel."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_base(model, state, initialization):
    """B loads only frozen CLIP weights; downstream remains exactly constructor-initialized."""
    if initialization == "warm_start_v0":
        model.load_state_dict(state, strict=True)
        return {"mode": initialization, "downstream_initial_sha256": downstream_digest(model)}
    if initialization != "fresh_downstream":
        raise ValueError(f"Unknown initialization: {initialization}")
    before = downstream_digest(model)
    clip_state = {name: value for name, value in state.items() if name.startswith("clipmodel.")}
    expected_clip = {name for name in model.state_dict() if name.startswith("clipmodel.")}
    if not expected_clip or set(clip_state) != expected_clip:
        raise ValueError("Released checkpoint has incomplete/incompatible frozen CLIP weights")
    result = model.load_state_dict(clip_state, strict=False)
    if result.unexpected_keys or any(name.startswith("clipmodel.") for name in result.missing_keys):
        raise ValueError("Failed to restore the frozen CLIP backbone")
    after = downstream_digest(model)
    if before != after:
        raise RuntimeError("Fresh downstream was changed while loading frozen CLIP")
    return {"mode": initialization, "constructor_downstream_sha256": before,
            "downstream_initial_sha256": after, "v0_downstream_loaded": False,
            "frozen_clip_source": "released V0 checkpoint; no downstream keys loaded"}


def validate_split_coverage(labels, train, dev, prompts, output, required=True):
    """Persist class support and stop before training if the validation task is incomplete."""
    counts = {part: (labels[indices] > 0).sum(0).tolist()
              for part, indices in (("train", train), ("development", dev))}
    missing = {part: [prompts[i] for i, count in enumerate(values) if count == 0]
               for part, values in counts.items()}
    report = {"prompt_order": prompts, "counts": counts, "missing_labels": missing,
              "required": required}
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if required and any(missing.values()):
        raise ValueError(
            f"Incomplete training/development class coverage: {missing}. "
            "No training started. Review the split and source exclusions; "
            "do not select a multi-class alignment checkpoint on an incomplete validation set.")
    return report


def video_input(array, width, device, rng=None, trace=None):
    """One stratified temporal sample <=width for train; all original steps for dev."""
    source_steps = len(array)
    indices = np.arange(source_steps)
    if rng is not None and len(array) > width:
        boundaries = np.linspace(0, len(array), width + 1).astype(int)
        indices = np.array([rng.integers(a, max(a + 1, b))
                            for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])
        array = array[indices]
    if trace is not None:
        trace.update(source_steps=source_steps, selected_indices=indices.tolist())
    steps = len(array)
    blocks = math.ceil(steps / width)
    values = torch.zeros((blocks * width, 512), dtype=torch.float32, device=device)
    values[:steps] = torch.as_tensor(array, dtype=torch.float32, device=device)
    lengths = (steps - torch.arange(blocks) * width).clamp(max=width)
    return values.reshape(blocks, width, 512), lengths, steps


def video_scores(c, a):
    """Label-free development scores; no ground-truth choice of top-k size."""
    k = topk_size(len(c), True)
    return (float(c.sigmoid().topk(k).values.mean()),
            float((1 - a.softmax(-1)[:, 0]).topk(k).values.mean()))


def train_preflight(model, pack, ids, labels, abnormal, train, args, prompts,
                    make_mask, device, loss_config, calibrate, calibrate_bias=False,
                    conditioning=False, probe_limit=32, health=False):
    """Train-only initialization/gradient probe; no optimizer update or test inputs."""
    selected = (probe_indices(train, labels, abnormal, probe_limit) if conditioning else
                np.r_[train[~abnormal[train]][:4], train[abnormal[train]][:4]])
    if len(selected) < 2:
        raise ValueError("Preflight requires normal and abnormal train videos")
    model.eval()
    norms = []
    with torch.no_grad():
        for index in selected:
            x, lengths, count = video_input(pack.array(pack.features[ids[int(index)]]),
                                           args.visual_length, device, np.random.default_rng(42))
            v = model.base.encode_video(x, make_mask(lengths, args.visual_length).to(device), lengths)
            norms.append(float(model.geometry.visual(v.float()).reshape(-1, v.shape[-1])[:count]
                               .norm(dim=-1).median()))
        t = model.base.encode_textprompt(prompts).float()
        text_norm = float(model.geometry.text(t).norm(dim=-1).median())
        visual_norm = float(np.median(norms))
        if not all(math.isfinite(n) and n > 1e-8 for n in (visual_norm, text_norm)):
            raise ValueError("Invalid train-only projector initialization norms")
        if conditioning:
            model.geometry.set_modality_norms(visual_norm, text_norm)
        elif calibrate:
            # Set median projected norm to 1 before the paper's lambda=0.7 scaling.
            model.geometry.visual.weight.div_(visual_norm)
            model.geometry.text.weight.div_(text_norm)
        if calibrate_bias:
            distances = []
            for index in selected:
                x, lengths, count = video_input(pack.array(pack.features[ids[int(index)]]),
                                               args.visual_length, device, np.random.default_rng(42))
                v = model.base.encode_video(x, make_mask(lengths, args.visual_length).to(device), lengths)
                logits = model.geometry.classify(model.geometry.map_visual(v))
                distances.append(-logits.reshape(-1)[:count] + model.geometry.bias)
            bias = torch.cat(distances).median()
            if not torch.isfinite(bias):
                raise ValueError("Non-finite train-only classifier bias calibration")
            model.geometry.bias.copy_(bias)
    rows, geometry_probes, health_probes = [], [], []
    health_ids = {int(selected[0]), int(selected[abnormal[selected]][0])}
    for index in selected:
        model.zero_grad(set_to_none=True)
        x, lengths, _ = video_input(pack.array(pack.features[ids[int(index)]]),
                                   args.visual_length, device, np.random.default_rng(42))
        _, c, a, aux = model.forward_with_aux(
            x, make_mask(lengths, args.visual_length).to(device), prompts, lengths)
        losses, parts = full_loss(c, a, labels[int(index)].to(device)[None], lengths,
                                  aux, "poincare", loss_config)
        if not bool(torch.isfinite(losses).all()):
            raise ValueError(f"Non-finite preflight loss: {ids[int(index)]}")
        grads = torch.autograd.grad(parts["bce"].sum(),
                                    (model.geometry.prototype, model.geometry.bias), retain_graph=True)
        if not all(bool(torch.isfinite(g).all()) for g in grads) or not any(
                bool(g.abs().sum() > 0) for g in grads):
            raise RuntimeError(f"Preflight BCE gradient failed for train video {ids[int(index)]}")
        if health and int(index) in health_ids:
            health_probes.append({"video_id": ids[int(index)], **learning_health(
                model, c, a, aux, parts, labels[int(index)].to(device)[None], lengths, loss_config)})
        losses.mean().backward()
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        if bad:
            raise RuntimeError(f"Preflight non-finite gradients: {ids[int(index)]}: {bad}")
        valid = aux["valid"]
        radius = aux["curvature"].sqrt() * aux["visual"].norm(dim=-1)[valid]
        text_radius = aux["curvature"].sqrt() * aux["text"].norm(dim=-1)
        cone_argument = loss_config["cone_aperture_scale"] * (1 - text_radius.square()) / (
            text_radius.clamp_min(1e-8))
        probe = {"video_id": ids[int(index)],
                 "visual_relative_radius_max": float(radius.detach().max()),
                 "text_relative_radius_max": float(text_radius.detach().max()),
                 "cone_valid_parent_fraction": float((cone_argument < 1).float().mean()),
                 "alignment_logit_min": float(a.detach().min()),
                 "alignment_logit_max": float(a.detach().max())}
        geometry_probes.append(probe)
        rows.append([ids[int(index)], f"{float(parts['bce'].detach().mean()):.5f}",
                     f"{float(c.detach()[valid].min()):.3f}",
                     f"{float(c.detach()[valid].max()):.3f}",
                     f"{float(radius.detach().max()):.6f}"])
    model.zero_grad(set_to_none=True)
    print(render_table(["Train probe video", "BCE", "C min", "C max", "max sqrt(c)|x|"],
                       rows, numeric=(1, 2, 3, 4)), flush=True)
    print("Train-only preflight passed: finite forward/backward, BCE classifier gradients nonzero.",
          flush=True)
    return {"videos": [ids[int(i)] for i in selected], "test_data_used": False,
            "optimizer_updates": 0, "calibrated": calibrate,
            "classifier_bias_calibrated": calibrate_bias,
            "classifier_bias": float(model.geometry.bias.detach()),
            "conditioning": conditioning, "learning_health": health_probes,
            "geometry_probes": geometry_probes,
            "visual_projected_median_before": visual_norm,
            "text_projected_median_before": text_norm, "probe_rows": rows}


def fit(base, pack, ids, labels, abnormal, train, dev, args, prompts,
        make_mask, smoke, config, geometry_config, output, seed, half_batch, device,
        prefix, initial_sha, native_loss=None, native_loss_receipt=None):
    data.seed_all(seed)
    native = config.get("native_cosine_control", False)
    if native and (native_loss is None or native_loss_receipt is None):
        raise ValueError("Native control requires the extracted pinned upstream objective")
    if native and any(key in config for key in ("two_stage", "from_scratch")):
        raise ValueError("Native control uses the matched joint recipe, not a different schedule")
    model = (NativePipeline(base) if native else FullPipeline(base, "poincare", geometry_config)).to(device)
    space = "native_cosine" if native else "poincare"
    schema = NATIVE_SCHEMA if native else SCHEMA
    objective = native_loss if native else full_loss
    health_fn = native_health if native else learning_health
    model_receipt = model.model_receipt if native else model.geometry.receipt
    checkpoint_extra = ({"native_objective": native_loss_receipt,
                         "native_forward_sha256": hashlib.sha256(
                             inspect.getsource(model._native_forward).encode()).hexdigest()}
                        if native else {})
    if "two_stage" in config and "from_scratch" in config:
        raise ValueError("Select exactly one training policy, A or B")
    policy = (FromScratchTraining(model, config, smoke) if "from_scratch" in config else
              TwoStageTraining(model, config, smoke) if "two_stage" in config else None)
    input_fn = repository_video_input if config.get("native_repository_recipe") else video_input
    batch_config = {"batch_size": 2 * half_batch, **config}
    optimizer = policy.optimizer if policy else joint_optimizer(model, config)
    epochs = policy.epochs if policy else (1 if smoke else config["epochs"])
    scheduler = (None if policy else
                 torch.optim.lr_scheduler.MultiStepLR(optimizer, config["scheduler_milestones"],
                                                       gamma=config["scheduler_gamma"])
                 if config.get("native_repository_recipe") else
                 torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs))
    if policy:
        policy.begin_epoch(0)
    amp = bool(config.get("amp", True) and device.type == "cuda")
    initial_scale = config.get("amp_initial_scale", 1.0)
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError("amp_initial_scale must be finite and positive")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=initial_scale)
    loss_config = {**config["loss"], "phi": config["phi"][prefix]}
    if config.get("single_temperature") and loss_config.get("alignment_loss_temperature", 1.) != 1.:
        raise ValueError("Single-temperature recipe must not rescale alignment logits inside CE")
    preflight = (native_preflight(model, pack, ids, labels, abnormal, train, args, prompts,
                                 make_mask, device, objective, loss_config, input_fn,
                                 ignore_padding_mask=config.get("native_repository_recipe", False))
                 if native else train_preflight(model, pack, ids, labels, abnormal, train, args, prompts,
                               make_mask, device, loss_config,
                               config.get("calibrate_initial_scale", True),
                               config.get("calibrate_classifier_bias", False),
                               conditioning=config.get("condition_modality_norms", False),
                               probe_limit=config.get("initialization_probe_videos", 32),
                                health=config.get("learning_health", False)))
    (output / "preflight.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    data.seed_all(seed)
    if config.get("verify_encoder_flow", False):
        # Probe the real pinned model, not the unit-test adapter. No optimizer
        # step or test labels are involved; retain observed call order and source.
        index = int(train[abnormal[train]][0])
        x, lengths, _ = input_fn(pack.array(pack.features[ids[index]]),
                                   args.visual_length, device, np.random.default_rng(seed))
        model.zero_grad(set_to_none=True)
        with EncoderFlowAudit(model) as audit:
            _, c, a, aux = model.forward_with_aux(
                x, (None if config.get("native_repository_recipe") else
                    make_mask(lengths, args.visual_length).to(device)), prompts, lengths)
            loss, _ = objective(c, a, labels[index].to(device)[None], lengths,
                                aux, space, loss_config)
            loss.mean().backward()
            flow = {**audit.receipt(), "video_id": ids[index]}
        (output / "encoder_flow.json").write_text(json.dumps(flow, indent=2), encoding="utf-8")
        model.zero_grad(set_to_none=True)
        print(("Encoder flow verified on train: native local/global, classifier FFN/FC, "
               "prompt -> frozen text encoder -> native fusion; gradients present."
               if native else
               "Encoder flow verified on train: local Transformer -> global GCN -> visual mapping; "
               "text prompts -> frozen text encoder -> text mapping; all adapter/prompt gradients present."),
              flush=True)
        data.seed_all(seed)
    history, health_history, best_ap, best_epoch = [], [], -1., None
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pth"
    final_epoch_only = config.get("data_protocol") == "official_train_final_epoch"
    if final_epoch_only and len(dev):
        raise ValueError("Official full-train mode must not select checkpoints on development")
    selection = "fixed_final_epoch" if final_epoch_only else (
        "classification" if prefix == "ucf" else "alignment")
    print(render_table(["Training setting", "Value"], [
        ["Epochs", epochs], ["Effective batch", batch_config["batch_size"]],
        ["Microbatch", config.get("microbatch_size", 1)], ["Optimizer", type(optimizer).__name__], ["Learning rate", config["learning_rate"]],
        ["Weight decay", config["weight_decay"]], ["Scheduler", "two-stage warmup/cosine" if policy else type(scheduler).__name__],
        ["Selection", "fixed final epoch; no dev/test selection" if final_epoch_only else f"development {selection} video AP"],
        ["AMP", amp], ["Initial loss scale", initial_scale],
        ["Max AMP retries per batch", config.get("amp_max_retries", 8)],
        *[[name, count] for name, count in model.parameter_groups().items()],
    ]), flush=True)
    planned_batches = len(recipe_batches(train, abnormal, batch_config, seed)) * epochs
    if smoke:
        planned_batches = min(2, planned_batches)
    started = time.perf_counter()
    first_gradient_groups = None
    checked_phases = set()
    for epoch in range(epochs):
        phase_receipt = policy.begin_epoch(epoch) if policy else {
            "phase": "joint_native_cosine_control" if native else
                     "joint_sanity_fixed_curvature" if config.get("sanity_fix") else "joint_legacy"}
        if policy:
            print("Training phase:", json.dumps(phase_receipt), flush=True)
        batches = recipe_batches(train, abnormal, batch_config, seed + epoch)
        if smoke:
            batches = batches[:2]
        rng = np.random.default_rng(seed + epoch)
        losses, updates, skipped = [], 0, 0
        components = {key: [] for key in (("bce", "alignment", "contrastive") if native else
                                         ("bce", "alignment", "entailment", "contrastive"))}
        if policy:
            policy.train_mode()
        else:
            model.train()
        for step, batch in enumerate(batches, 1):
            def backward():
                total_loss = 0.
                batch_health = []
                batch_components = {key: [] for key in components}
                microbatch = config.get("microbatch_size", 1)
                if type(microbatch) is not int or microbatch < 1:
                    raise ValueError("microbatch_size must be a positive integer")
                for position in range(0, len(batch), microbatch):
                    indices = batch[position:position + microbatch]
                    sample_traces = [{} for _ in indices]
                    inputs = [input_fn(pack.array(pack.features[ids[int(index)]]),
                                          args.visual_length, device, rng, trace=sample_trace)
                              for index, sample_trace in zip(indices, sample_traces, strict=True)]
                    visual = torch.cat([item[0] for item in inputs], dim=0)
                    lengths = torch.cat([item[1] for item in inputs])
                    target = labels[indices].to(device)
                    detailed = (config.get("learning_health") and
                                step <= (config.get("health_first_epoch_batches", 2) if epoch == 0 else
                                         config.get("health_batches_per_epoch", 2)))
                    # Hooks are observational; they make no extra stochastic training forward.
                    stage_trace = StageTrace(model.base, getattr(model, "geometry", None)) if detailed else None
                    try:
                        if stage_trace is not None:
                            stage_trace.__enter__()
                        with torch.autocast(device_type=device.type, enabled=amp):
                            _, c, a, aux = model.forward_with_aux(
                                visual, (None if config.get("native_repository_recipe") else
                                         make_mask(lengths, args.visual_length).to(device)), prompts, lengths)
                    except Exception as exc:
                        append_json(output / "training_flow.jsonl", {
                            "epoch": epoch+1, "batch": step, "video_ids": [ids[int(i)] for i in indices],
                            "status": "failed", "stage": stage_trace.active_stage if stage_trace else "forward",
                            "error": f"{type(exc).__name__}: {exc}"})
                        raise
                    finally:
                        if stage_trace is not None:
                            stage_trace.__exit__()
                    try:
                        with torch.autocast(device_type=device.type, enabled=False):
                            per_video, parts = objective(c, a, target, lengths, aux, space, loss_config)
                            loss = per_video.mean()
                        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(v).all()) for v in parts.values()):
                            raise ValueError("Nonfinite native/Poincare loss component")
                    except Exception as exc:
                        append_json(output / "training_flow.jsonl", {
                            "epoch": epoch+1, "batch": step, "video_ids": [ids[int(i)] for i in indices],
                            "status": "failed", "stage": "loss", "error": f"{type(exc).__name__}: {exc}"})
                        raise
                    if config.get("audit_data_flow", True):
                        try:
                            flow_row = check_batch(visual, lengths, target, c, a, aux,
                                                   [ids[int(i)] for i in indices], sample_traces)
                            append_json(output / "training_flow.jsonl", {
                                "epoch": epoch+1, "batch": step, "microbatch_position": position,
                                "status": "ok", **flow_row,
                                "internal_stages": stage_trace.rows if stage_trace is not None else [],
                                "internal_stages_probed": stage_trace is not None,
                                "loss_components": {k: float(v.detach().mean()) for k,v in parts.items()}})
                        except Exception as exc:
                            append_json(output / "training_flow.jsonl", {
                                "epoch": epoch+1, "batch": step, "video_ids": [ids[int(i)] for i in indices],
                                "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                            raise
                    if detailed:
                        batch_health.append({"video_ids": [ids[int(i)] for i in indices],
                            **health_fn(model, c, a, aux, parts, target, lengths, loss_config)})
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite {space} loss: epoch={epoch+1}, "
                                         f"videos={[ids[int(i)] for i in indices]}")
                    weight = len(indices) / len(batch)
                    scaler.scale(loss * weight).backward()
                    total_loss += float(loss.detach()) * weight
                    for key in batch_components:
                        batch_components[key].extend(parts[key].detach().reshape(-1).tolist())
                return {"loss": total_loss, "components": batch_components, "health": batch_health}

            def before_step():
                nonlocal first_gradient_groups
                if any(p.grad is not None for p in model.base.clipmodel.parameters()):
                    raise RuntimeError("Frozen CLIP received parameter gradients")
                if policy:
                    receipt = policy.check_gradients(require_nonzero=policy.phase not in checked_phases)
                    if policy.phase not in checked_phases:
                        checked_phases.add(policy.phase)
                        first_gradient_groups = model.gradient_receipt()
                        if policy.phase != "head_warmup" and not first_gradient_groups["prompt_tokens"]:
                            raise RuntimeError("Joint fine-tuning lost prompt-token gradients")
                        print("Phase gradient check:", json.dumps(receipt), flush=True)
                elif first_gradient_groups is None:
                    first_gradient_groups = model.gradient_receipt()
                    for group in ("temporal_adapter", "prompt_fusion", "prompt_tokens",
                                  "classification_head" if native else "geometry_classifier"):
                        if not first_gradient_groups[group]:
                            raise RuntimeError(f"Disconnected training gradient: {group}")

            try:
                payload, update = optimizer_update(
                    model, optimizer, scaler, backward, amp=amp,
                    gradient_clip=config["gradient_clip"], retry_limit=config.get("amp_max_retries", 8),
                    rng=rng, before_step=before_step,
                    diagnostics_path=output / "gradient_diagnostics.json")
            except Exception as exc:
                append_json(output / "optimizer_flow.jsonl", {
                    "epoch": epoch+1, "batch": step, "video_ids": [ids[int(i)] for i in batch],
                    "status": "failed", "stage": "forward_backward_or_optimizer_update",
                    "error": f"{type(exc).__name__}: {exc}"})
                raise
            total_loss = payload["loss"]
            if config.get("audit_data_flow", True):
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad):
                    raise ValueError(f"Nonfinite parameters after update: epoch={epoch+1}, batch={step}")
                append_json(output / "optimizer_flow.jsonl", {
                    "epoch": epoch+1, "batch": step, "video_ids": [ids[int(i)] for i in batch],
                    "finite_parameters_after_update": True, "frozen_clip_has_gradients": False,
                    "group_learning_rates": [g["lr"] for g in optimizer.param_groups], **update})
            if payload["health"]:
                health_history.append({"epoch": epoch + 1, "batch": step,
                                       "optimizer_update": update, "microbatches": payload["health"]})
                (output / "learning_health.json").write_text(
                    json.dumps(health_history, indent=2), encoding="utf-8")
                probe = payload["health"][0]
                print("Train health:", json.dumps(probe if native else {
                    "epoch": epoch + 1, "batch": step, "radii": probe["radii"],
                    "cone": probe["cone"],
                    "entail_visual_gradient": probe["loss_gradients"]["entailment"]["visual"],
                }), flush=True)
            for key in components:
                components[key].extend(payload["components"][key])
            skipped += update["overflow_retries"]
            updates += 1
            losses.append(total_loss)
            if epoch == 0 and step == 1:
                measured = time.perf_counter() - started
                print(f"{space}: first batch {measured:.2f}s; rough train-only estimate "
                      f"{measured * planned_batches / 60:.1f} min (excludes development/test)",
                      flush=True)
            if step % 25 == 0 or step == len(batches):
                print(f"{space}: epoch {epoch + 1}/{epochs}, batch {step}/{len(batches)}, "
                      f"loss={total_loss:.5f}, updates={updates}, amp_retries={skipped}, "
                      f"loss_scale={scaler.get_scale():g}", flush=True)
        if updates == 0:
            raise RuntimeError("Epoch completed without a single finite optimizer update")
        lr = max(group["lr"] for group in optimizer.param_groups)
        if scheduler is not None:
            scheduler.step()
        model.eval()
        predictions = []
        with torch.no_grad():
            for index in dev:
                array = pack.array(pack.features[ids[int(index)]])
                visual, lengths, count = video_input(array, args.visual_length, device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    _, c, a = model(visual, make_mask(lengths, args.visual_length).to(device),
                                    prompts, lengths)
                predictions.append(video_scores(c.reshape(-1)[:count],
                                                a.reshape(-1, len(prompts))[:count]))
        aps = ([None, None] if final_epoch_only else [
            float(average_precision_score(abnormal[dev], np.array(predictions)[:, j])) for j in (0, 1)])
        ap = None if final_epoch_only else aps[0 if prefix == "ucf" else 1]
        if ap is not None and not math.isfinite(ap):
            raise ValueError("Non-finite development AP")
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "dev_classification_video_ap": aps[0], "dev_alignment_video_ap": aps[1],
               "components": {key: float(np.mean(value)) for key, value in components.items()},
               "learning_rate": lr, "next_learning_rate": (None if policy else optimizer.param_groups[0]["lr"]),
               "optimizer_updates": updates, "amp_skipped_updates": skipped,
               "amp_loss_scale": scaler.get_scale(),
                "geometry": model_receipt(), "training_phase": phase_receipt}
        history.append(row)
        if (final_epoch_only and epoch == epochs - 1) or (ap is not None and ap > best_ap):
            best_ap, best_epoch = ap, epoch + 1
            save_checkpoint({"schema": schema, "space": space, "geometry_config": geometry_config,
                        "loss_config": loss_config, "initial_v0_sha256": initial_sha,
                        "epoch": epoch + 1, "downstream": model.downstream_state(),
                        "train_config": config, "training_phase": phase_receipt,
                        **checkpoint_extra}, checkpoint)
        if epoch + 1 in config.get("checkpoint_epochs", []):
            save_checkpoint({"schema": schema, "space": space, "geometry_config": geometry_config,
                             "loss_config": loss_config, "initial_v0_sha256": initial_sha,
                             "epoch": epoch + 1, "downstream": model.downstream_state(),
                             "train_config": config, "training_phase": phase_receipt,
                             **checkpoint_extra}, output / f"epoch_{epoch+1:03d}.pth")
        # Keep optimizer/RNG state locally for recovery; excluded from the compact ZIP.
        save_checkpoint({"schema": schema, "epoch": epoch + 1, "downstream": model.downstream_state(),
                    "optimizer": optimizer.state_dict(), "scheduler": policy.state_dict() if policy else scheduler.state_dict(),
                    "training_phase": phase_receipt,
                    "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "config": config, "geometry_config": geometry_config,
                    "initial_v0_sha256": initial_sha, **checkpoint_extra}, output / "last_training_state.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(render_table(["Epoch", "Loss", "Dev C AP %", "Dev A AP %", "LR", "Updates", "AMP retries"],
                           [[epoch + 1, f"{row['train_loss']:.5f}", ("N/A" if aps[0] is None else f"{aps[0]*100:.4f}"),
                             ("N/A" if aps[1] is None else f"{aps[1]*100:.4f}"), f"{lr:.3g}", updates, skipped]],
                           numeric=tuple(range(7))), flush=True)
        print(f"Checkpoint rule {selection}; saved epoch: {best_epoch}", flush=True)
        if native:
            print(render_table(["Native BCE", "Native alignment", "Native contrastive (raw)"],
                               [[f"{row['components'][key]:.5f}" for key in components]],
                               numeric=(0, 1, 2)), flush=True)
        else:
            print(render_table(["BCE", "Alignment", "Entailment", "Separation", "c", "lambda_v", "lambda_t"],
                           [[*[f"{row['components'][key]:.5f}" for key in components],
                             f"{row['geometry']['curvature_magnitude']:.5f}",
                             f"{row['geometry']['lambda_visual']:.5f}",
                             f"{row['geometry']['lambda_text']:.5f}"]],
                           numeric=tuple(range(7))), flush=True)
    audit_summary = (summarize_training(output, health_history, [ids[int(i)] for i in train])
                     if config.get("audit_data_flow", True) else None)
    receipt = {
        "space": space, "checkpoint_file": str(checkpoint.relative_to(output.parent)),
        "checkpoint_sha256": data.file_sha(checkpoint), "best_epoch": best_epoch,
        "best_dev_primary_video_ap": best_ap, "selection_branch": selection, "history": history,
        "parameter_groups": model.parameter_groups(), "first_gradient_groups": first_gradient_groups,
        "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "elapsed_seconds": time.perf_counter() - started, "config": config,
        "training_policy": policy.mode if policy else (
            "native_cosine_control" if native else
            "sanity_fixed_curvature" if config.get("sanity_fix") else "legacy_joint"),
        "geometry_config": geometry_config,
        "trained_scope": ("native classifier/adapter/prompt/fusion; only CLIP frozen" if native else
                          "all downstream including temporal adapter, prompt, CLS; only CLIP frozen"),
        "initial_v0_sha256": initial_sha, "effective_batch": batch_config["batch_size"],
        "preflight": preflight,
        "data_flow_audit": audit_summary,
        "learning_health_file": f"{output.name}/learning_health.json" if config.get("learning_health") else None,
        "model_padding_mask": "None (native trainer)" if config.get("native_repository_recipe") else "length mask",
        "microbatch": config.get("microbatch_size", 1), "amp": amp, "optimizer": type(optimizer).__name__,
        "optimizer_groups": [{"weight_decay": g["weight_decay"], "betas": list(g["betas"]),
                              "parameter_count": sum(p.numel() for p in g["params"])}
                             for g in optimizer.param_groups],
        "scheduler": "phase_warmup_then_cosine" if policy else type(scheduler).__name__,
        **checkpoint_extra,
    }
    return receipt


def run_training(config):
    output = Path(config["OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    data.SRC_REPO = Path(config["SRC_REPO"])
    data.checkout()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU; original VadCLIP forward requires CUDA")
    prefix, seed, smoke = config["PREFIX"], config["SEED"], config["SMOKE_ONLY"]
    v0_path = Path(config["V0_PATH"])
    if data.file_sha(v0_path) != config["V0_SHA256"]:
        raise ValueError("Released V0 checkpoint hash mismatch")
    for name in ("ftfy", "regex", "tqdm", "pandas"):
        if importlib.util.find_spec(name) is None:
            subprocess.run([sys.executable, "-m", "pip", "install", name], check=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    supplied_clip = config.get("CLIP_WEIGHTS")
    clip_candidates = [Path(supplied_clip)] if supplied_clip else sorted(
        Path(config["INPUT"]).rglob("ViT-B-16.pt"))
    if clip_candidates:
        clip_weight = next((p for p in clip_candidates if data.file_sha(p) == data.CLIP_SHA), None)
        if clip_weight is None:
            raise ValueError("Supplied CLIP ViT-B-16.pt SHA-256 mismatch")
        cache = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.resolve() != clip_weight.resolve():
            shutil.copy2(clip_weight, cache)
    with tempfile.TemporaryDirectory(prefix="vadclip_v21_") as temporary:
        runtime = Path(temporary) / "src"
        shutil.copytree(data.SRC_REPO / "src", runtime)
        clip_file = runtime / "clip" / "clip.py"
        source = clip_file.read_text(encoding="utf-8")
        marker = "from pkg_resources import packaging"
        if source.count(marker) != 1:
            raise ValueError("Unexpected pinned CLIP source")
        clip_file.write_text(source.replace(marker, "import packaging.version"), encoding="utf-8")
        sys.path.insert(0, str(runtime))
        from model import CLIPVAD
        from utils.tools import get_batch_label, get_batch_mask, get_prompt_text

        args = importlib.import_module(f"{prefix}_option").parser.parse_args([])
        train_rows, train_sha = data.official_csv(args.train_list, data.SRC_REPO)
        test_rows, test_sha = data.official_csv(args.test_list, data.SRC_REPO)
        label_map = {name: name.lower() for name in data.UCF_CLASSES} if prefix == "ucf" else data.XD_LABELS
        if prefix == "ucf":
            label_map["RoadAccidents"] = "roadAccidents"
        prompts = get_prompt_text(label_map)
        if len(prompts) != args.classes_num or prompts[0].casefold() != "normal":
            raise ValueError("Invalid prompt order")

        def encode(labels):
            return get_batch_label(labels, prompts, label_map)

        full_crops = config["TRAIN_CONFIG"].get("native_repository_recipe", False)
        if full_crops:
            resolved_root = config["TRAIN_ROOT"]
            if resolved_root is None:
                crop_receipt = Path(config["CROP_INVENTORY"]).with_name("preflight.json")
                resolved_root = json.loads(crop_receipt.read_text(encoding="utf-8"))["resolved_train_root"]
            pack = FullCropPack(resolved_root, prefix, train_rows)
            pack.load_inventory(config["CROP_INVENTORY"])
        else:
            pack = data.FeaturePack(Path(config["TRAIN_ROOT"] or config["INPUT"]), prefix,
                                    config["MANIFEST_SHA256"])
        try:
            if not full_crops:
                train_rows = data.select_train_rows(train_rows, pack.features, encode)
            split = data.dataset_split(train_rows, test_rows, pack.features, encode, prefix, seed,
                                       config["TRAIN_CONFIG"].get("data_protocol", "source_disjoint_development"),
                                       preserve_crops=full_crops)
            ids, groups, labels, abnormal, train, dev, excluded, overlap = split
            manifest = [{"id": ids[int(i)], "group": groups[int(i)], "split": part,
                         "label": train_rows[int(i)]["label"],
                         "feature_original": pack.features[ids[int(i)]]["original"],
                         "feature_steps": pack.features[ids[int(i)]]["steps"],
                         "sha256": pack.features[ids[int(i)]]["sha256"]}
                        for part, indices in (("train", train), ("development", dev),
                                              ("excluded_source_name", excluded)) for i in indices]
            manifest_path = output / "split_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if config["TRAIN_CONFIG"].get("data_protocol") == "official_train_final_epoch":
                expected_count = (1610 if prefix == "ucf" else 3954) * (10 if full_crops else 1)
                if len(train) != expected_count:
                    raise ValueError(f"Official {prefix} train expected {expected_count} videos, got {len(train)}")
                counts = (labels[train] > 0).sum(0).tolist()
                if any(count == 0 for count in counts):
                    raise ValueError("Official training split is missing classes")
                (output / "split_health.json").write_text(json.dumps({
                    "protocol": "official_train_final_epoch", "train_counts": counts,
                    "development_selection": False, "source_overlap_groups": overlap,
                    "excluded_videos": 0}, indent=2), encoding="utf-8")
            else:
                validate_split_coverage(labels, train, dev, prompts, output / "split_health.json",
                                        config["TRAIN_CONFIG"].get("require_class_coverage", True))
            print(f"{prefix}: train={len(train)}, dev={len(dev)}, excluded={len(excluded)}, "
                  f"source groups overlapping official test before filter={overlap}", flush=True)
            print(f"Initialization={config['TRAIN_CONFIG'].get('initialization', 'warm_start_v0')}; "
                  f"data_protocol={config['TRAIN_CONFIG'].get('data_protocol', 'source_disjoint_development')}; "
                  f"source overlap groups={overlap}, excluded clips={len(excluded)}.",
                  flush=True)
            if smoke:
                dev = np.r_[dev[~abnormal[dev]][:2], dev[abnormal[dev]][:2]]
            data.seed_all(seed)
            model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                            args.visual_head, args.visual_layers, args.attn_window,
                            args.prompt_prefix, args.prompt_postfix, device).to(device)
            state = torch.load(v0_path, map_location="cpu", weights_only=True)
            for field in ("state_dict", "model_state_dict"):
                if isinstance(state, dict) and field in state:
                    state = state[field]
                    break
            initialization = initialize_base(
                model, state, config["TRAIN_CONFIG"].get("initialization", "warm_start_v0"))
            print("Initialization:", json.dumps(initialization), flush=True)
            (output / "initialization.json").write_text(
                json.dumps(initialization, indent=2), encoding="utf-8")
            del state
            native = config["TRAIN_CONFIG"].get("native_cosine_control", False)
            variant = "cosine_control" if native else "v21"
            native_loss, native_loss_receipt = None, None
            if native:
                original_trainer = data.SRC_REPO / "src" / f"{prefix}_train.py"
                native_loss, native_loss_receipt = native_objective(
                    original_trainer.read_text(encoding="utf-8"), str(original_trainer))
                (output / "native_objective.json").write_text(
                    json.dumps(native_loss_receipt, indent=2), encoding="utf-8")
            variants = {variant: fit(
                model, pack, ids, labels, abnormal, train, dev, args, prompts,
                get_batch_mask, smoke, config["TRAIN_CONFIG"], config["GEOMETRY_CONFIG"],
                output / variant, seed, (config["HALF_BATCH"] if (native or config["TRAIN_CONFIG"].get("paper_design"))
                                      else 2 if smoke else config["HALF_BATCH"]), device,
                prefix, config["V0_SHA256"], native_loss, native_loss_receipt,
            )}
            if not smoke and len(pack.checked) != len(train_rows) - len(excluded):
                raise RuntimeError("Some eligible train/development features were not hash-verified")
        finally:
            pack.close()
    report = {"mode": config.get("RUN_MODE", "smoke" if smoke else "full"), "prefix": prefix, "seed": seed,
              "variants": variants, "initialization": initialization, "train_count": len(train), "development_count": len(dev),
              "prompt_order": prompts, "label_map": label_map,
              "development_evaluated_count": len(dev), "excluded_count": len(excluded),
              "source_overlap_before_filter": overlap, "train_csv_sha256": train_sha,
              "test_csv_sha256": test_sha, "split_manifest_sha256": data.file_sha(manifest_path),
              "feature_manifest_sha256": pack.manifest_sha256, "verified_features": len(pack.checked),
              "initial_v0_sha256": config["V0_SHA256"],
              "selection": ("fixed_final_epoch" if config["TRAIN_CONFIG"].get("data_protocol") ==
                            "official_train_final_epoch" else "development primary video AP"),
              "test_labels_used_for_selection": False, "official_test_previously_seen": True,
              "reproduction_status": ("pinned repository recipe with explicit RNG/audit differences; not checkpoint history"
                                       if config["TRAIN_CONFIG"].get("native_repository_recipe") else
                                       "native VadCLIP forward/objective; matched V21 training recipe"
                                      if native else "full downstream Poincare adaptation; not exact author reproduction"),
              "cosine_control_trained": native}
    (output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        run_training(globals())
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc(), "status": "failed"}
        failure_dir = Path(globals().get("OUTPUT", "."))
        try:
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "failure.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8")
        except OSError as save_error:
            print(f"Could not save failure.json: {save_error}", file=sys.stderr)
        raise
