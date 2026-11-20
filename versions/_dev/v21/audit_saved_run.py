"""Replay saved V21 predictions and inspect split coverage without training or retuning."""

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def alignment_log_odds(logits, normal_index=0):
    """Rank-equivalent to 1-softmax(normal) before rounding; requires original logits.

    Diagnostic readout only. Does not reconstruct rankings from saved rounded scores.
    """
    if (logits.ndim < 1 or logits.shape[-1] < 2
            or not 0 <= normal_index < logits.shape[-1]
            or not bool(torch.isfinite(logits).all())):
        raise ValueError("Expected finite logits with normal and at least one abnormal class")
    values = logits.double() if logits.dtype == torch.float64 else logits.float()
    abnormal = torch.cat((values[..., :normal_index], values[..., normal_index + 1:]), dim=-1)
    return torch.logsumexp(abnormal, dim=-1) - values[..., normal_index]


def score_diagnostics(scores, frame_gt):
    scores, gt = np.asarray(scores), np.asarray(frame_gt).reshape(-1)
    if (scores.ndim != 2 or len(gt) != len(scores) * 16 or not len(scores)
            or not np.isfinite(scores).all() or not np.isin(gt, [0, 1]).all()
            or len(np.unique(gt)) != 2):
        raise ValueError("Invalid score/ground-truth arrays for the locked 16-frame grid")
    result = []
    for column in range(scores.shape[1]):
        values = scores[:, column]
        frames = np.repeat(values, 16)
        result.append({
            "auc_pct": float(100 * roc_auc_score(gt, frames)),
            "ap_pct": float(100 * average_precision_score(gt, frames)),
            "zero_fraction": float(np.mean(values == 0)),
            "one_fraction": float(np.mean(values == 1)),
            "near_zero_fraction": float(np.mean(values < 1e-6)),
            "near_one_fraction": float(np.mean(values > 1 - 1e-6)),
            "unique_scores": len(np.unique(values)), "snippets": len(values),
        })
    return result


def split_coverage(rows, dataset):
    names = ("Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
             "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing",
             "Vandalism") if dataset == "ucf" else ("A", "B1", "B2", "B4", "B5", "B6", "G")
    parts = {}
    for part in sorted({row["split"] for row in rows}):
        subset = [row for row in rows if row["split"] == part]
        counter = Counter()
        for row in subset:
            labels = {row["label"]} if dataset == "ucf" else set(row["label"].split("-"))
            counter.update(labels & set(names))
        parts[part] = {"videos": len(subset), "source_groups": len({r["group"] for r in subset}),
                       "label_counts": {name: counter[name] for name in names},
                       "missing_labels": [name for name in names if not counter[name]]}
    return parts


def audit(path, check_crc=True):
    result = {"archive": str(path), "scope": "saved-score replay and diagnostics; no model selection"}
    with ZipFile(path) as archive:
        if check_crc:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Archive CRC failed: {bad}")
        names = archive.namelist()
        summaries = [name for name in names if name.endswith("/summary.json")]
        if len(summaries) != 1:
            raise ValueError("Expected exactly one run summary")
        root = summaries[0].removesuffix("summary.json")
        summary = json.loads(archive.read(summaries[0]))
        result.update(run=root, crc_checked=check_crc, train_config=summary["train_config"],
                      geometry_config=summary["geometry_config"], datasets={})
        for dataset in ("ucf", "xd"):
            entry = result["datasets"][dataset] = {}
            arrays = {}
            for variant in ("v0", "v21"):
                matches = [name for name in names if name.startswith(root + dataset + "/" + variant + "/")
                           and name.endswith("/predictions.npz")]
                if len(matches) != 1:
                    raise ValueError(f"Expected one predictions file: {dataset}/{variant}")
                with np.load(io.BytesIO(archive.read(matches[0])), allow_pickle=False) as data:
                    arrays[variant] = {key: data[key] for key in data.files}
                values = arrays[variant]
                entry[variant] = dict(zip(values["score_columns"].tolist(), score_diagnostics(
                    values["scores"], values["frame_gt"]), strict=True))
            for key in arrays["v0"].keys() - {"scores"}:
                if not np.array_equal(arrays["v0"][key], arrays["v21"][key]):
                    raise ValueError(f"V0/V21 evaluation grid mismatch: {dataset}/{key}")
            rows = json.loads(archive.read(root + dataset + "/train/split_manifest.json"))
            entry["split"] = split_coverage(rows, dataset)
            training = json.loads(archive.read(root + dataset + "/train/training.json"))
            record = training["variants"]["v21"]
            entry["initialization"] = training.get("initialization")
            entry["best_epoch"] = record["best_epoch"]
            entry["best_development_video_ap"] = record["best_dev_primary_video_ap"]
            history = record["history"]
            entry["first_epoch"] = history[0]
            entry["last_epoch"] = history[-1]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved verified replay:", args.output)
