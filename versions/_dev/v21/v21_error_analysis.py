"""Post-hoc benchmark failures. Never imported by the training subprocess.

The fixed 0.5 threshold is an uncalibrated diagnostic, not selected on the test.
Ranking metrics and raw-vs-rounded scores accompany it. No automatic causal claim.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("dataset,variant,branch,video_id\n", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def ranking_tail(y, score, fraction=.01):
    """Exact fractional membership at a tied top-fraction cutoff; diagnostic only."""
    y, score = np.asarray(y, dtype=bool), np.asarray(score)
    if not len(y) or y.shape != score.shape or not 0 < fraction <= 1:
        raise ValueError("Invalid ranking-tail inputs")
    budget = len(y) * fraction
    cutoff = float(np.partition(score, len(score) - int(np.ceil(budget)))[-int(np.ceil(budget))])
    above, tied = score > cutoff, score == cutoff
    tie_weight = (budget - int(above.sum())) / int(tied.sum())
    weights = above.astype(float) + tied * tie_weight
    return {"top_fraction": fraction, "cutoff": cutoff, "tie_weight": float(tie_weight),
            "precision": float(weights[y].sum() / budget),
            "false_positive_frame_weight": float(weights[~y].sum()),
            "ties_at_cutoff": int(tied.sum())}, weights


def analyze_ranking_tail(dataset, paths, output):
    """Saved predictions only: score tails, worst background segments and video mass."""
    ranking, distributions, segments, videos = [], [], [], []
    for variant, path in paths.items():
        report, arrays = load_prediction(path)
        fpf = report["frames_per_feature"]
        gt = arrays["frame_gt"].reshape(-1).astype(bool)
        normal = np.repeat(np.array([str(x).casefold() in {"a", "normal"}
                                    for x in arrays["video_label"]]),
                           np.diff(arrays["snippet_offsets"]) * fpf)
        for col, branch in enumerate(arrays["score_columns"].astype(str)):
            score = np.repeat(arrays["scores"][:, col], fpf)
            info, weights = ranking_tail(gt, score)
            ranking.append({"dataset": dataset, "variant": variant, "branch": branch, **info})
            for name, mask in (("normal_video_background", normal & ~gt),
                               ("abnormal_video_background", ~normal & ~gt), ("event", gt)):
                values = score[mask]
                q = np.quantile(values, [0, .5, .9, .99, .999, 1]) if len(values) else [None] * 6
                distributions.append({"dataset": dataset, "variant": variant, "branch": branch,
                                      "group": name, "frames": len(values),
                                      "mean": float(values.mean()) if len(values) else None,
                                      **dict(zip(("min", "median", "p90", "p99", "p999", "max"), q, strict=True))})
            for i, video in enumerate(arrays["ids"]):
                lo, hi = arrays["snippet_offsets"][i:i+2] * fpf
                local_score, local_gt = score[lo:hi], gt[lo:hi]
                fp_weight = weights[lo:hi] * ~local_gt
                if not bool(fp_weight.any()):
                    continue
                common = {"dataset": dataset, "variant": variant, "branch": branch,
                          "video_id": video, "label": str(arrays["video_label"][i])}
                videos.append({**common, "top1pct_false_positive_frame_weight": float(fp_weight.sum())})
                for start, end in intervals((weights[lo:hi] > 0) & ~local_gt):
                    segments.append({**common, "start_frame_0based": int(start),
                                     "end_frame_exclusive": int(end), "frames": int(end-start),
                                     "top1pct_false_positive_frame_weight": float(fp_weight[start:end].sum()),
                                     "score_mean": float(local_score[start:end].mean()),
                                     "score_max": float(local_score[start:end].max()),
                                     "observed_stage": "cross_video_high_score_background_tail",
                                     "causal_origin": "unresolved; inspect raw-output and upstream evidence"})
    videos.sort(key=lambda x: x["top1pct_false_positive_frame_weight"], reverse=True)
    segments.sort(key=lambda x: (x["score_max"], x["top1pct_false_positive_frame_weight"]), reverse=True)
    write_csv(Path(output) / "ranking_tail_summary.csv", ranking)
    write_csv(Path(output) / "score_distributions.csv", distributions)
    write_csv(Path(output) / "top1pct_false_positive_videos.csv", videos)
    write_csv(Path(output) / "top_false_positive_segments.csv", segments)
    return {"ranking_tail_summary": ranking,
            "scope": "test-only descriptive diagnostics; cutoff never enters training or model selection"}


def load_prediction(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    with np.load(Path(path).with_name("predictions.npz"), allow_pickle=False) as data:
        arrays = dict(data)
    id_key = "video_id" if "video_id" in arrays else "official_path"
    ids = arrays[id_key].astype(str)
    offsets = arrays["snippet_offsets"]
    scores, gt = arrays["scores"], arrays["frame_gt"].reshape(-1)
    fpf = report["frames_per_feature"]
    if (type(fpf) is not int or fpf < 1 or len(ids) != len(set(ids)) or
            len(arrays["video_label"]) != len(ids) or offsets.shape != (len(ids) + 1,) or
            offsets.dtype.kind not in "iu" or offsets[0] != 0 or
            bool((np.diff(offsets) <= 0).any()) or offsets[-1] != len(scores) or
            scores.shape != (offsets[-1], 2) or
            not np.isfinite(scores).all() or bool(((scores < 0) | (scores > 1)).any()) or
            len(gt) != offsets[-1] * fpf or not np.isin(gt, [0, 1]).all()):
        raise ValueError("Invalid saved ID/offset/score/frame-GT grid")
    arrays["ids"] = ids
    return report, arrays


def probability_replay(trace, directory):
    """Check saved device probabilities against an independent FP64 CPU reference.

    CPU and CUDA reductions are not bitwise interchangeable. Use a disclosed
    absolute budget in probability space based on the stored logit precision
    and class count. Never replace the probabilities used for benchmark metrics.
    """
    z = torch.from_numpy(trace["classification_logits"])
    align = torch.from_numpy(trace["alignment_logits"])
    if z.ndim != 1 or align.ndim != 2 or align.shape[1] < 2:
        raise ValueError("Invalid raw diagnostic logit shapes")
    if any(trace[key].dtype not in (np.float32, np.float64) for key in ("classification_logits", "alignment_logits")):
        raise ValueError("Replay requires stored FP32/FP64 logits")
    eps = [np.finfo(trace[key].dtype).eps for key in ("classification_logits", "alignment_logits")]
    tolerances = [float(max(8*eps[0], 64*np.finfo(np.float64).eps)),
                  float(max((align.shape[1]+4)*eps[1], 64*np.finfo(np.float64).eps))]
    reference = torch.stack((z.double().sigmoid(), 1-align.double().softmax(-1)[:, 0]), -1).numpy()
    error = np.abs(reference-trace["scores"])
    rows, examples = [], []
    for column, branch in enumerate(("classification", "alignment")):
        bad = np.flatnonzero(error[:, column] > tolerances[column])
        rows.append({"branch": branch, "absolute_tolerance": tolerances[column],
                     "max_absolute_error": float(error[:, column].max()),
                     "p99_absolute_error": float(np.quantile(error[:, column], .99)),
                     "mismatch_count": len(bad), "stored_logit_dtype": str((z, align)[column].dtype)})
        for token in sorted(bad, key=lambda i: error[i, column], reverse=True)[:10]:
            video = int(np.searchsorted(trace["snippet_offsets"], token, side="right")-1)
            examples.append({"video_id": str(trace["video_id"][video]), "branch": branch,
                             "snippet_index": int(token-trace["snippet_offsets"][video]),
                             "recorded_probability": float(trace["scores"][token, column]),
                             "reference_probability_fp64": float(reference[token, column]),
                             "absolute_error": float(error[token, column]),
                             "raw_logits": (float(z[token]) if column == 0 else align[token].tolist())})
    receipt = {"schema": "v21_probability_replay_v2", "reference": "CPU float64 from stored raw logits",
               "policy": "absolute probability error: C 8*eps; A (classes+4)*eps; floor 64*eps64",
               "status": "failed" if examples else "passed", "branches": rows, "examples": examples,
               "benchmark_scores_modified": False,
               "scope": "cross-backend numerical check; ID/offset/export/GT/metric checks are separate"}
    (Path(directory) / "probability_replay.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    if examples:
        worst = examples[0]
        raise ValueError(f"Saved raw logits do not reproduce trace probabilities beyond dtype budget: "
                         f"{worst['video_id']}/{worst['branch']}/snippet={worst['snippet_index']}; "
                         f"abs_error={worst['absolute_error']:.9g}; see probability_replay.json")
    return receipt


def load_trace(path, arrays):
    directory = Path(path).parent / "diagnostics"
    with np.load(directory / "diagnostic_tensors.npz", allow_pickle=False) as data:
        trace = dict(data)
    trace_ids, offsets = trace["video_id"].astype(str), trace["snippet_offsets"]
    if len(trace_ids) != len(set(trace_ids)) or set(trace_ids) != set(arrays["ids"]):
        raise ValueError("Trace/evaluator video identities differ")
    if (offsets.shape != (len(trace_ids) + 1,) or offsets[0] != 0 or
            offsets.dtype.kind not in "iu" or bool((np.diff(offsets) <= 0).any())):
        raise ValueError("Invalid trace offsets")
    keys = ("classification_logits", "alignment_logits", "alignment_log_odds", "scores",
            "adapter_norm", "feature_norm", "block_index")
    for key in keys:
        if len(trace[key]) != offsets[-1] or not np.isfinite(trace[key]).all():
            raise ValueError(f"Invalid diagnostic tensor: {key}")
    if trace["scores"].shape != arrays["scores"].shape:
        raise ValueError("Diagnostic score columns differ from evaluator")
    align = torch.from_numpy(trace["alignment_logits"])
    probability_replay(trace, directory)
    odds = (torch.logsumexp(align.double()[:, 1:], -1)-align.double()[:, 0]).numpy()
    if not np.allclose(odds, trace["alignment_log_odds"], rtol=1e-10, atol=1e-8):
        raise ValueError("Saved alignment log odds differ from raw alignment logits")
    index = {video_id: i for i, video_id in enumerate(trace_ids)}
    aligned = {key: [] for key in keys}
    for i, video_id in enumerate(arrays["ids"]):
        j = index[video_id]
        start, end = offsets[j:j+2]
        left, right = arrays["snippet_offsets"][i:i+2]
        if end - start != right - left:
            raise ValueError(f"Trace/evaluator per-video lengths differ: {video_id}")
        for key in keys:
            aligned[key].append(trace[key][start:end])
    aligned = {key: np.concatenate(value) for key, value in aligned.items()}
    if not np.allclose(aligned["scores"], arrays["scores"], rtol=1e-6, atol=1e-7):
        raise ValueError("Raw-logit score conversion/trim differs from benchmark export")
    return aligned


def analyze_benchmark(dataset, paths, output, require_trace=True):
    """Compare every treatment to V0 with the same per-video frame grid and hashes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if "v0" not in paths or len(paths) < 2:
        raise ValueError("Error analysis requires same-run V0 and a treatment")
    all_rows, errors, events, evidence = [], [], [], []
    sources, signature, feature_reference = {}, None, None
    for variant, path in paths.items():
        report, arrays = load_prediction(path)
        sources[variant] = {"run_json": str(path), "prediction_sha256": hashlib.sha256(
            Path(path).with_name("predictions.npz").read_bytes()).hexdigest()}
        current_signature = (arrays["ids"].tolist(), arrays["snippet_offsets"].tolist(),
                             arrays["video_label"].tolist(), arrays["score_columns"].tolist(),
                             report["frames_per_feature"], report["visual_block_length"],
                             hashlib.sha256(arrays["frame_gt"].tobytes()).hexdigest())
        if signature is not None and signature != current_signature:
            raise ValueError("V0/treatment sample/label/score/frame grids differ")
        signature = current_signature
        videos = {str(v.get("id", v.get("official_path"))): v for v in report["videos"]}
        if len(videos) != len(arrays["ids"]) or set(videos) != set(arrays["ids"]):
            raise ValueError("Input metadata/evaluator video identities differ")
        feature_hashes = {key: value.get("feature_sha256") for key, value in videos.items()}
        if any(not value for value in feature_hashes.values()):
            raise ValueError("Missing per-video input feature provenance")
        if feature_reference is not None and feature_hashes != feature_reference:
            raise ValueError("V0/treatment per-video input feature hashes differ")
        feature_reference = feature_hashes
        trace_path = Path(path).parent / "diagnostics/diagnostic_tensors.npz"
        trace = load_trace(path, arrays) if require_trace or trace_path.exists() else None
        fpf, width = report["frames_per_feature"], report["visual_block_length"]
        replay_metrics = {}
        for column, branch in enumerate(arrays["score_columns"].astype(str)):
            y, score = arrays["frame_gt"], np.repeat(arrays["scores"][:, column], fpf)
            replay_metrics[branch] = {"roc_auc": auc(y, score),
                                      "ap": float(average_precision_score(y, score))}
            saved = report["metrics"][branch]
            if "ano_auc" in saved:
                abnormal = np.array([str(label).casefold() not in {"a", "normal"}
                                     for label in arrays["video_label"]])
                mask = np.repeat(abnormal, np.diff(arrays["snippet_offsets"])*fpf)
                replay_metrics[branch]["ano_auc"] = auc(y[mask], score[mask])
            for metric, value in replay_metrics[branch].items():
                saved_key = "frame_ap" if metric == "ap" and "frame_ap" in saved else metric
                if saved_key in saved and (value is None or abs(value - saved[saved_key]) > 1e-9):
                    raise ValueError(f"Saved benchmark metric cannot be replayed: {variant}/{branch}/{metric}")
            if trace is not None:
                raw = trace["classification_logits"] if column == 0 else trace["alignment_log_odds"]
                raw = np.repeat(raw, fpf)
                replay_metrics[branch].update(raw_roc_auc=auc(y, raw),
                                             raw_ap=float(average_precision_score(y, raw)))
        sources[variant]["replayed_metrics"] = replay_metrics
        for i, video_id in enumerate(arrays["ids"]):
            start, end = (int(x) for x in arrays["snippet_offsets"][i:i+2])
            label = str(arrays["video_label"][i])
            y = arrays["frame_gt"][start*fpf:end*fpf].astype(bool)
            normal = label.casefold() in {"a", "normal"}
            if normal and bool(y.any()):
                raise ValueError(f"Normal video contains event frame GT: {video_id}")
            if videos[video_id]["steps"] != end - start:
                raise ValueError(f"Input feature/benchmark step count differs: {video_id}")
            if trace is not None and not np.array_equal(trace["block_index"][start:end], np.arange(end-start)//width):
                raise ValueError(f"Trace block boundaries differ from evaluator grid: {video_id}")
            gt_events = intervals(y)
            for col, branch in enumerate(arrays["score_columns"].astype(str)):
                snippet_score = arrays["scores"][start:end, col]
                score = np.repeat(snippet_score, fpf)
                predicted = score >= .5
                fp, fn = predicted & ~y, ~predicted & y
                raw = (trace["classification_logits"] if col == 0 else trace["alignment_log_odds"])[start:end] if trace else None
                raw_frame = np.repeat(raw, fpf) if raw is not None else None
                rank_auc = auc(y, score)
                raw_auc = auc(y, raw_frame) if raw_frame is not None else None
                stage = ("probability_rounding" if raw_auc is not None and rank_auc is not None and raw_auc - rank_auc > 1e-6
                         else "raw_output_ranking" if raw_auc is not None and raw_auc < .5
                         else "probability_output_ranking_no_raw_trace" if raw is None and rank_auc is not None and rank_auc < .5
                         else "fixed_threshold_decision" if (fp.any() or fn.any())
                         else "no_error_at_fixed_threshold")
                raw_diagnostic = "available" if trace is not None else "unavailable_historical_output"
                k = 1 + (end-start) // 16
                top = np.argsort(-snippet_score, kind="stable")[:k]
                token_event_fraction = y.reshape(-1, fpf).mean(-1)
                norm = trace["adapter_norm"][start:end] if trace else None
                row = {"dataset": dataset, "variant": variant, "branch": branch, "video_id": video_id,
                       "label": label, "feature_sha256": feature_hashes[video_id],
                       "steps": end-start, "frames": len(y), "block_count": (end-start+width-1)//width,
                       "padding_fraction": 1-(end-start)/(((end-start+width-1)//width)*width),
                       "event_frames": int(y.sum()), "event_fraction": float(y.mean()),
                       "event_count": len(gt_events), "shortest_event_frames": min((b-a for a,b in gt_events), default=0),
                       "within_video_auc": rank_auc, "raw_within_video_auc": raw_auc,
                       "frame_ap": float(average_precision_score(y, score)) if y.any() else None,
                       "tp_frames": int((predicted & y).sum()), "fp_frames": int(fp.sum()),
                       "fn_frames": int(fn.sum()), "tn_frames": int((~predicted & ~y).sum()),
                       "event_recall_at_0_5": float(predicted[y].mean()) if y.any() else None,
                       "background_fpr_at_0_5": float(predicted[~y].mean()) if (~y).any() else None,
                       "event_background_score_gap": float(score[y].mean()-score[~y].mean()) if y.any() and (~y).any() else None,
                       "score_std": float(score.std()), "exact_zero_one_fraction": float(((score==0)|(score==1)).mean()),
                       "diagnostic_topk": k, "topk_event_fraction": float(token_event_fraction[top].mean()),
                       "adapter_norm_mean": float(norm.mean()) if norm is not None else None,
                       "observed_stage": stage, "raw_trace": raw_diagnostic,
                       "causal_origin": "unresolved; upstream stages may cause the observed output error"}
                all_rows.append(row)
                if fp.any() or fn.any() or (rank_auc is not None and rank_auc < .5):
                    evidence.append({"dataset": dataset, "variant": variant, "branch": branch,
                                     "video_id": video_id, "observed_stage": stage,
                                     "evidence_level": "observed score behavior; causal origin unresolved",
                                     "probability_auc": rank_auc, "raw_auc": raw_auc,
                                     "fp_frames_at_0_5": int(fp.sum()), "fn_frames_at_0_5": int(fn.sum())})
                for kind, mask in (("FP", fp), ("FN", fn)):
                    for lo, hi in intervals(mask):
                        left, right = lo//fpf, (hi-1)//fpf+1
                        errors.append({"dataset": dataset, "variant": variant, "branch": branch,
                                       "video_id": video_id, "label": label, "error": kind,
                                       "start_frame_0based": int(lo), "end_frame_exclusive": int(hi),
                                       "frames": int(hi-lo), "start_snippet": int(left), "end_snippet_exclusive": int(right),
                                       "start_block": int(left//width), "end_block": int((right-1)//width),
                                       "score_mean": float(score[lo:hi].mean()),
                                       "raw_score_mean": float(raw_frame[lo:hi].mean()) if raw_frame is not None else None,
                                       "adapter_norm_mean": float(norm[left:right].mean()) if norm is not None else None,
                                       "observed_stage": stage})
                for lo, hi in gt_events:
                    events.append({"dataset": dataset, "variant": variant, "branch": branch,
                                   "video_id": video_id, "label": label, "start_frame_0based": int(lo),
                                   "end_frame_exclusive": int(hi), "frames": int(hi-lo),
                                   "recall_at_0_5": float(predicted[lo:hi].mean()),
                                   "fully_missed_at_0_5": not bool(predicted[lo:hi].any()),
                                   "score_mean": float(score[lo:hi].mean()),
                                   "crosses_block_boundary": lo//(width*fpf) != (hi-1)//(width*fpf)})
    indexed = {(r["variant"], r["branch"], r["video_id"]): r for r in all_rows}
    paired = []
    for row in all_rows:
        if row["variant"] == "v0":
            continue
        ref = indexed["v0", row["branch"], row["video_id"]]
        paired.append({"dataset": dataset, "variant": row["variant"], "branch": row["branch"],
                       "video_id": row["video_id"], "label": row["label"],
                       "v0_within_video_auc": ref["within_video_auc"], "treatment_within_video_auc": row["within_video_auc"],
                       "auc_delta_pp": 100*(row["within_video_auc"]-ref["within_video_auc"]) if row["within_video_auc"] is not None else None,
                       "fp_frame_delta_at_0_5": row["fp_frames"]-ref["fp_frames"],
                       "fn_frame_delta_at_0_5": row["fn_frames"]-ref["fn_frames"],
                       "observed_stage": row["observed_stage"]})
    grouped = defaultdict(list)
    for row in all_rows:
        attributes = {"label": row["label"], "length": "multiple_blocks" if row["block_count"] > 1 else "one_block",
                      "event_density": "no_event" if row["event_fraction"] == 0 else "sparse_le_10pct" if row["event_fraction"] <= .1 else "dense_gt_10pct",
                      "short_event": "no_event" if row["event_count"] == 0 else "le_64_frames" if row["shortest_event_frames"] <= 64 else "gt_64_frames",
                      "rounding": "ge_95pct_exact_0_or_1" if row["exact_zero_one_fraction"] >= .95 else "lt_95pct_exact_0_or_1"}
        for name, value in attributes.items():
            grouped[row["variant"], row["branch"], name, value].append(row)
    characteristics = []
    for (variant, branch, name, value), group in grouped.items():
        eligible = [r["within_video_auc"] for r in group if r["within_video_auc"] is not None]
        positives, negatives = sum(r["event_frames"] for r in group), sum(r["frames"]-r["event_frames"] for r in group)
        characteristics.append({"dataset": dataset, "variant": variant, "branch": branch,
                                "characteristic": name, "value": value, "video_count": len(group),
                                "eligible_auc_videos": len(eligible), "macro_within_video_auc": float(np.mean(eligible)) if eligible else None,
                                "fp_frames_at_0_5": sum(r["fp_frames"] for r in group),
                                "fn_frames_at_0_5": sum(r["fn_frames"] for r in group),
                                "event_recall_at_0_5": sum(r["tp_frames"] for r in group)/positives if positives else None,
                                "background_fpr_at_0_5": sum(r["fp_frames"] for r in group)/negatives if negatives else None,
                                "interpretation": "descriptive association; groups overlap; no causal claim"})
    for name, rows in (("per_video_errors.csv", all_rows), ("error_segments.csv", errors),
                       ("event_diagnostics.csv", events), ("paired_video_errors.csv", paired),
                       ("characteristic_summary.csv", characteristics), ("stage_evidence.csv", evidence)):
        write_csv(output / name, rows)
    summary = {"schema": "v21_flow_error_audit_v1", "dataset": dataset, "sources": sources,
               "diagnostic_threshold": .5, "threshold_source": "fixed before test; uncalibrated; diagnostic only",
               "frame_intervals": "0-based [start,end); no assumed FPS",
               "test_annotations_scope": "post-hoc only; no model/checkpoint/threshold selection",
               "checks_passed": ["unique video IDs", "per-video feature hashes", "same V0/treatment input grid",
                                  "frame expansion", "binary GT", "normal label consistency", "metric replay"],
               "trace_required": require_trace,
               "video_branch_rows": len(all_rows), "error_segment_count": len(errors),
               "stage_evidence_count": len(evidence),
               "causal_diagnosis": "unresolved; observations identify where errors are visible, not the first causal stage",
               "hypotheses_to_test": [
                   {"if": "raw rank exceeds rounded-probability rank", "hypothesis": "score conversion loses temporal ordering",
                    "check": "report raw-logit ranks with fixed checkpoint; separate stable native MIL-loss ablation"},
                   {"if": "ranking already weak in raw outputs", "hypothesis": "upstream representation, head or MIL supervision may fail temporal discrimination",
                    "check": "hold recipe fixed; ablate one adapter/fusion/pooling component at a time on development data"},
                   {"if": "multiple-block or short-event groups regress", "hypothesis": "temporal context or sampling may be limiting",
                    "check": "inspect saved sample indices/block boundaries; controlled train/dev exposure or window experiment"}],
               "limits": ["cached features cannot establish scene appearance, lighting or camera motion",
                          "training traces and test traces refer to different videos; no per-test-sample training causality",
                          "GT-dependent characteristics are descriptive only; not model inputs"]}
    if require_trace:
        summary["checks_passed"].append("raw forward replay / score conversion / trim matches benchmark export")
    summary["ranking_tail"] = analyze_ranking_tail(dataset, paths, output)
    (output / "error_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    worst = sorted([r for r in paired if r["auc_delta_pp"] is not None], key=lambda r: r["auc_delta_pp"])[:12]
    text = [f"# Chẩn đoán mẫu sai: {dataset}", "",
            "Ngưỡng 0.5 cố định chỉ dùng thống kê FP/FN; chưa hiệu chỉnh xác suất. Xem thêm AUC theo video và raw logits.",
            "Khoảng frame dùng chỉ số từ 0, đầu đóng/cuối mở. Không quy đổi giây khi chưa có FPS.", "",
            "| Video | Nhánh | Δ AUC so V0 (đpt) | Bước quan sát lỗi |",
            "|---|---|---:|---|"]
    text.extend(f"| {r['video_id'].split('/')[-1]} | {r['branch']} | {r['auc_delta_pp']:+.2f} | {r['observed_stage']} |" for r in worst)
    text += ["", "Tất cả video: per_video_errors.csv. Đoạn FP/FN: error_segments.csv. Sự kiện bỏ sót: event_diagnostics.csv.",
             "So V0: paired_video_errors.csv. Đặc điểm nhóm: characteristic_summary.csv. Bằng chứng theo bước: stage_evidence.csv.",
              "Đuôi top 1%: ranking_tail_summary.csv. Phân phối điểm: score_distributions.csv. FP cao nhất: top_false_positive_segments.csv và top1pct_false_positive_videos.csv.",
             "", "Đã quan sát đầu ra sai ở bước ghi trong bảng; nguồn gây lỗi có thể nằm phía trước. Chưa chứng minh nguyên nhân gốc.",
             "Không suy diễn nội dung hình ảnh từ tên file hoặc norm embedding. Xem error_summary.json để chọn đối chứng tiếp theo trên train/dev."]
    (output / "ERROR_ANALYSIS.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return summary
