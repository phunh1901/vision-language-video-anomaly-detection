"""Paired, video-bootstrap metrics without treating correlated frames as iid."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranked_counts(scores, positives, negatives):
    order = np.argsort(-scores, kind="stable")
    end = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(order) - 1]
    return order, end, positives[order], negatives[order]


def weighted_metrics(ranked, video_weights):
    order, end, pos, neg = ranked
    weights = video_weights[order]
    tp = np.cumsum(pos * weights)[end]
    fp = np.cumsum(neg * weights)[end]
    if not len(tp) or tp[-1] <= 0 or fp[-1] <= 0:
        return None
    recall = tp / tp[-1]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    ap = np.sum(np.diff(np.r_[0, recall]) * precision)
    x, y = np.r_[0, fp / fp[-1]], np.r_[0, recall]
    auc = np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2)
    return float(auc), float(ap)


def paired_bootstrap(scores0, scores1, gt, offsets, repeats, seed=42):
    positives = gt.reshape(-1, 16).sum(1).astype(float)
    negatives = 16 - positives
    ranks = [ranked_counts(s, positives, negatives) for s in (scores0, scores1)]
    video_index = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        counts = rng.multinomial(len(offsets) - 1, np.full(len(offsets) - 1, 1 / (len(offsets) - 1)))
        values = [weighted_metrics(r, counts[video_index]) for r in ranks]
        if all(v is not None for v in values):
            deltas.append(np.subtract(values[1], values[0]) * 100)
    if not deltas:
        return {"requested": repeats, "valid": 0, "auc_ci95_pp": None, "ap_ci95_pp": None}
    bounds = np.quantile(deltas, [0.025, 0.975], axis=0)
    return {"requested": repeats, "valid": len(deltas),
            "auc_ci95_pp": bounds[:, 0].tolist(), "ap_ci95_pp": bounds[:, 1].tolist()}


def false_alarm_diagnostics(scores, gt, offsets):
    """Retrospective test-label operating points, NOT deployable calibrated thresholds."""
    frames = np.repeat(scores, 16)
    truth = gt.astype(bool)
    positive_video = np.zeros(len(gt), dtype=bool)
    for a, b in zip(offsets[:-1] * 16, offsets[1:] * 16, strict=True):
        positive_video[a:b] = truth[a:b].any()
    result = []
    for requested in (0.5, 0.8):
        positive_scores = np.sort(frames[truth])[::-1]
        threshold = float(positive_scores[max(0, int(np.ceil(requested * len(positive_scores))) - 1)])
        detected = frames >= threshold
        row = {"requested_recall": requested, "actual_recall": float(detected[truth].mean()),
               "threshold": threshold, "status": "retrospective_test_diagnostic_only"}
        for name, mask in (("all_background", ~truth),
                           ("background_in_positive_videos", ~truth & positive_video),
                           ("all_negative_videos", ~truth & ~positive_video)):
            row[name + "_frames"] = int(mask.sum())
            row[name + "_fp"] = int((detected & mask).sum())
            row[name + "_fpr"] = float(detected[mask].mean()) if mask.any() else None
        result.append(row)
    return result


def score_arrays(scores, gt):
    x = np.repeat(scores, 16)
    return {"roc_auc": float(roc_auc_score(gt, x)),
            "ap": float(average_precision_score(gt, x))}
