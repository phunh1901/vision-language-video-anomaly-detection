"""Label-free runtime traces; frame annotations are read only by post-hoc analysis.

No tensors here change a forward, loss, gradient, optimizer or sample selection.
Diagnostic reruns happen after the timed inference call with the model in eval mode.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch


def append_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def tensor_summary(value):
    value = value.detach()
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError("Empty/nonfinite tensor in observed data flow")
    x = value.double()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "min": float(x.min()), "max": float(x.max()),
            "rms": float(x.square().mean().sqrt())}


class StageTrace:
    """Temporary hooks on real modules; output statistics include padded tokens."""

    def __init__(self, base, geometry=None):
        modules = dict(base.named_children())
        names = [n for n in modules if n in {"temporal", "linear", "classifier", "mlp1", "mlp2",
                                            "text_prompt_embeddings"} or
                 n.startswith("gc") and n[2:].isdigit()]
        self.modules = {n: modules[n] for n in names}
        if hasattr(base, "clipmodel") and hasattr(base.clipmodel, "transformer"):
            self.modules["clip_text_transformer"] = base.clipmodel.transformer
        if geometry is not None:
            self.modules.update(visual_projection=geometry.visual, text_projection=geometry.text)
        self.rows, self.handles, self.adapter = [], [], None
        self.active_stage = "input"

    def __enter__(self):
        for name, module in self.modules.items():
            def before(module, args, name=name):
                self.active_stage = name

            def after(module, args, value, name=name):
                tensors = [value] if isinstance(value, torch.Tensor) else list(value)
                info = [tensor_summary(x) for x in tensors if isinstance(x, torch.Tensor)]
                self.rows.append({"stage": name, "outputs": info})
                if name == "linear":
                    self.adapter = value.detach()

            self.handles.extend((module.register_forward_pre_hook(before),
                                 module.register_forward_hook(after)))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


def check_batch(visual, lengths, targets, c, a, aux, ids, samples):
    """Checks every training microbatch, including identity/temporal sampling receipts."""
    sizes = torch.as_tensor(lengths, device=visual.device)
    b, width, dim = visual.shape
    if (b != len(ids) or dim != 512 or len(samples) != b or sizes.shape != (b,)
            or bool(((sizes < 1) | (sizes > width)).any())):
        raise ValueError(f"Invalid train sample/length/grid: {ids}")
    if c.shape != (b, width, 1) or a.shape[:2] != (b, width) or targets.shape != (b, a.shape[-1]):
        raise ValueError(f"Train logits/targets do not match input grid: {ids}")
    if (not bool(((targets == 0) | (targets == 1)).all()) or
            bool((targets.sum(-1) < 1).any()) or
            bool(((targets[:, 0] > 0) & (targets[:, 1:].sum(-1) > 0)).any())):
        raise ValueError(f"Invalid encoded normal/anomaly targets: {ids}")
    valid = torch.arange(width, device=visual.device)[None] < sizes[:, None]
    if not torch.equal(valid, aux["valid"]):
        raise ValueError("Training padding/valid mask mismatch")
    stages = {"sampled_input": tensor_summary(visual), "adapter": tensor_summary(aux["adapter"]),
              "classification_logits": tensor_summary(c), "alignment_logits": tensor_summary(a)}
    if bool((visual[~valid] != 0).any()):
        raise ValueError("Nonzero training padding")
    rows = []
    for i, (video_id, sample) in enumerate(zip(ids, samples, strict=True)):
        n = int(sizes[i])
        indices = np.asarray(sample["selected_indices"])
        if (len(indices) != n or indices.min() < 0 or indices.max() >= sample["source_steps"]
                or bool((np.diff(indices) <= 0).any())):
            raise ValueError(f"Invalid temporal sample indices: {video_id}")
        ranges = sample.get("pooling_ranges")
        if ranges is not None:
            edges = np.linspace(0, sample["source_steps"], width + 1, dtype=np.int32)
            expected = np.stack([edges[:-1], edges[1:]], axis=1)
            if n != width or not np.array_equal(ranges, expected):
                raise ValueError(f"Invalid uniform pooling ranges: {video_id}")
        z = c[i, :n, 0].detach().float()
        p = z.sigmoid()
        abnormal = bool(targets[i, 0] == 0)
        rows.append({"video_id": video_id, **sample, "valid_steps": n,
                     "abnormal_video_label": abnormal,
                     "classification_logit_min": float(z.min()),
                     "classification_logit_max": float(z.max()),
                     "classification_probability_mean": float(p.mean()),
                     "rounded_zero_count": int((p == 0).sum()),
                     "rounded_one_count": int((p == 1).sum()),
                     "video_label_opposite_rounded_count": int((p == (0 if abnormal else 1)).sum()),
                     "adapter_norm_mean": float(aux["adapter"][i, :n].detach().norm(dim=-1).mean())})
    return {"stage_checks": stages, "videos": rows}


def summarize_training(output, health, expected_ids=None):
    """Actual sample exposure, not assumed coverage of an event without frame labels."""
    output = Path(output)
    coverage, rows_checked, wrong_rounding = {}, 0, 0
    path = output / "training_flow.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["status"] != "ok":
                continue
            rows_checked += 1
            for video in row["videos"]:
                key = video["video_id"]
                n = video["source_steps"]
                if key not in coverage:
                    coverage[key] = {"seen": np.zeros(n, dtype=bool), "exposures": 0,
                                     "opposite_rounded_tokens": 0, "valid_tokens": 0}
                entry = coverage[key]
                if len(entry["seen"]) != n:
                    raise ValueError(f"Source feature length changed during training: {key}")
                if video.get("pooling_ranges") is not None:
                    for lo, hi in video["pooling_ranges"]:
                        entry["seen"][lo:hi] = True
                else:
                    entry["seen"][video["selected_indices"]] = True
                entry["exposures"] += 1
                entry["opposite_rounded_tokens"] += video["video_label_opposite_rounded_count"]
                entry["valid_tokens"] += video["valid_steps"]
                wrong_rounding += video["video_label_opposite_rounded_count"]
    rows = [{"video_id": key, "source_steps": len(value["seen"]),
             "observed_forward_exposures": value["exposures"],
             "unique_sampled_steps": int(value["seen"].sum()),
             "sampled_step_coverage": float(value["seen"].mean()),
             "valid_tokens_observed": value["valid_tokens"],
             "video_label_opposite_rounded_tokens": value["opposite_rounded_tokens"]} for key,value in coverage.items()]
    if rows:
        with (output / "training_sample_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    alerts = []
    for batch in health:
        for micro in batch["microbatches"]:
            for video_id, entry in zip(micro["video_ids"], micro["temporal"], strict=True):
                if entry.get("wrong_saturated_pool"):
                    alerts.append({"epoch": batch["epoch"], "batch": batch["batch"], "video_id": video_id,
                                   "stage": "native_sigmoid_MIL_BCE", "evidence": "pooled probability rounded to wrong endpoint",
                                   "bce_gradient_at_raw_c_norm": entry.get("bce_gradient_at_raw_c_norm"),
                                   "causal_origin": "unresolved; compare LR-only pilot before any loss change"})
    summary = {"microbatch_checks": rows_checked, "unique_training_videos_observed": len(rows),
               "eligible_videos_not_observed": sorted(set(expected_ids or []) - set(coverage)),
               "video_label_opposite_rounded_tokens": wrong_rounding, "alerts": alerts,
               "rounding_count_scope": "opposite video-level target; not per-frame error labels for abnormal videos",
               "frame_labels_used": False, "diagnostic_changes_training": False,
               "coverage_scope": "observed training forwards, including AMP retries if any; not event coverage",
               "internal_stage_scope": "first configured health batches; input/adapter/logits/masks checked each microbatch",
               "training_sample_coverage_file": "training_sample_coverage.csv"}
    (output / "training_diagnosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class EvaluationTrace:
    def __init__(self, output, dataset, variant):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.dataset, self.variant = dataset, variant
        self.ids, self.parts, self.offsets, self.prompts = [], [], [0], None

    @torch.no_grad()
    def observe(self, video_id, base, dispatch, visual, lengths, mask, measured, steps, geometry=None, prompts=None,
                expected_features=None):
        trace = StageTrace(base, geometry)
        try:
            if base.training:
                raise ValueError("Diagnostic replay requires model.eval()")
            if visual.ndim != 3:
                raise ValueError("Evaluator input must be [blocks,width,features]")
            width = visual.shape[1]
            capacity = visual.shape[0]*width
            if not 0 < steps <= capacity:
                raise ValueError(f"Original feature steps exceed evaluator tensor capacity: {steps}/{capacity}")
            # Export always trims the original feature grid. Native model lengths
            # are a separate argument; XD's upstream implementation may include
            # padded context. Never replace those arguments or change its forward.
            valid = (torch.arange(capacity, device=visual.device) < steps).reshape(visual.shape[:2])
            export_lengths = valid.sum(-1)
            sizes = None if lengths is None else torch.as_tensor(lengths, device=visual.device)
            trace.active_stage = "native_context_metadata"
            if sizes is not None and (not bool(torch.isfinite(sizes).all()) or bool((sizes < 0).any())):
                raise ValueError("Nonfinite or negative native evaluator lengths")
            if mask is not None and (not bool(torch.isfinite(mask).all()) or
                                     not bool(((mask == 0) | (mask == 1)).all())):
                raise ValueError("Evaluator mask is nonfinite or not binary")
            length_contract = (sizes is not None and sizes.shape == (len(visual),)
                               and bool(torch.isfinite(sizes).all())
                               and bool(((sizes >= 0) & (sizes <= width)).all())
                               and torch.equal(sizes, export_lengths.to(sizes)))
            padding_count = int((visual[~valid] != 0).sum())
            mask_contract = mask is None or (mask.shape == valid.shape and torch.equal(mask.bool(), ~valid))
            if self.dataset != "xd":
                if not length_contract:
                    raise ValueError("Evaluator valid lengths do not match original feature steps")
                if not mask_contract:
                    raise ValueError("Evaluator padding mask is inconsistent with lengths")
                if padding_count:
                    raise ValueError("Nonzero evaluation padding")
            elif expected_features is None:
                raise ValueError("XD trace requires original features to verify exported token identity")
            trace.active_stage = "input_feature_identity"
            if expected_features is not None:
                original = np.asarray(expected_features)
                if original.shape != (steps, visual.shape[-1]) or not np.isfinite(original).all():
                    raise ValueError("Original feature shape/finite values differ from evaluator inventory")
                expected = torch.as_tensor(np.array(original, copy=True), device=visual.device, dtype=visual.dtype)
                if not torch.equal(visual.reshape(-1, visual.shape[-1])[:steps], expected):
                    raise ValueError("Evaluator input token values/order differ from original features")
            input_info = tensor_summary(visual)
            trace.active_stage = "native_forward_replay"
            with trace:
                replayed = dispatch()
            trace.active_stage = "raw_forward_outputs"
            if len(replayed) != len(measured):
                raise ValueError("Diagnostic/timed result structure mismatch")
            for name, left, right in zip(("text_encoder_output", "classification_logits", "alignment_logits"),
                                         measured, replayed, strict=True):
                trace.active_stage = name
                tensor_summary(left)
                tensor_summary(right)
                if left.shape != right.shape or not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
                    raise ValueError("Diagnostic replay differs from timed output")
            # Preserve the actual timed logits. Hooked replay may use a different
            # Transformer kernel with tiny floating-point differences.
            _, c, a = measured
            trace.active_stage = "output_grid_and_prompt_order"
            if c.shape != (*visual.shape[:2], 1) or a.shape[:2] != visual.shape[:2] or a.shape[-1] < 2:
                raise ValueError("Evaluation raw logits do not match block grid")
            if prompts is None or len(prompts) != a.shape[-1] or str(prompts[0]).casefold() != "normal":
                raise ValueError("Evaluation prompt order must match logits with normal at index 0")
            if self.prompts is not None and list(prompts) != self.prompts:
                raise ValueError("Prompt order changed between evaluation videos")
            self.prompts = list(prompts)
            if trace.adapter is None or trace.adapter.shape[:2] != visual.shape[:2]:
                raise ValueError("Missing temporal adapter output on evaluation grid")
            z = c.reshape(-1)[:steps]
            align = a.reshape(-1, a.shape[-1])[:steps]
            trace.active_stage = "probability_conversion"
            scores = torch.stack((z.sigmoid(), 1 - align.softmax(-1)[:, 0]), -1)
            odds = torch.logsumexp(align.double()[:, 1:], -1) - align.double()[:, 0]
            values = {"classification_logits": z, "alignment_logits": align,
                      "alignment_log_odds": odds, "scores": scores,
                      "adapter_norm": trace.adapter.norm(dim=-1).reshape(-1)[:steps],
                      "feature_norm": visual.norm(dim=-1).reshape(-1)[:steps],
                      "block_index": torch.arange(steps, device=z.device) // width}
            self.parts.append({k: v.cpu().numpy() for k, v in values.items()})
            self.ids.append(str(video_id))
            self.offsets.append(self.offsets[-1] + steps)
            append_json(self.output / "evaluation_flow.jsonl", {
                "video_id": str(video_id), "status": "ok", "input": input_info,
                "valid_lengths": export_lengths.cpu().tolist(), "export_steps": steps,
                "model_lengths": None if sizes is None else sizes.cpu().tolist(),
                "model_lengths_shape": None if sizes is None else list(sizes.shape),
                "model_lengths_match_export": bool(length_contract),
                "padding_mask_supplied": mask is not None, "mask_matches_export": bool(mask_contract),
                "model_mask_shape": None if mask is None else list(mask.shape),
                "nonzero_context_values_after_export": padding_count,
                "input_matches_original_features": expected_features is not None,
                "native_arguments_modified": False,
                "length_scope": "native model context and exported feature grid are separate; differences are observations",
                "normal_class_index": 0, "prompt_order": self.prompts,
                "stages": trace.rows, "classification_logits": tensor_summary(z),
                "alignment_logits": tensor_summary(align), "replay_matches_timed_output": True,
                "timing_scope": "diagnostic replay excluded from latency.measure"})
        except Exception as exc:
            append_json(self.output / "evaluation_flow.jsonl", {
                "video_id": str(video_id), "status": "failed", "stage": trace.active_stage,
                "original_steps": int(steps), "input_shape": list(visual.shape),
                "model_lengths_repr": repr(lengths),
                "completed_stages": trace.rows, "error": f"{type(exc).__name__}: {exc}"})
            raise

    def finish(self):
        if len(self.ids) != len(set(self.ids)) or not self.parts:
            raise ValueError("Empty or duplicated evaluation trace")
        np.savez_compressed(self.output / "diagnostic_tensors.npz",
                            video_id=np.asarray(self.ids), snippet_offsets=np.asarray(self.offsets),
                            **{k: np.concatenate([part[k] for part in self.parts]) for k in self.parts[0]})
        (self.output / "trace_contract.json").write_text(json.dumps({
            "schema": "v21_flow_error_audit_v1", "dataset": self.dataset, "variant": self.variant,
            "video_count": len(self.ids), "snippet_count": self.offsets[-1],
            "prompt_order": self.prompts, "normal_class_index": 0,
            "test_labels_used": False, "changes_predictions": False,
            "tolerances": {"replay_rtol": 1e-5, "replay_atol": 1e-6},
            "limitation": "shape/finite checks cannot establish semantic feature quality or causal blame",
        }, indent=2), encoding="utf-8")
