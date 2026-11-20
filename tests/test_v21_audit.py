"""Independent replay examples, split coverage and precision readout checks."""

import numpy as np
import pytest
import torch
from audit_saved_run import alignment_log_odds, score_diagnostics, split_coverage
from train_v21 import validate_split_coverage


def test_replay_matches_hand_ranked_auc_ap():
    scores = np.array([[.1], [.4], [.35], [.8]])
    gt = np.repeat([0, 0, 1, 1], 16)
    result = score_diagnostics(scores, gt)[0]
    assert result["auc_pct"] == pytest.approx(75)
    assert result["ap_pct"] == pytest.approx(100 * (1 + 2/3) / 2)


@pytest.mark.parametrize("scores,gt", [
    (np.array([[float("nan")]]), np.zeros(16)),
    (np.array([[.1]]), np.zeros(15)),
    (np.array([[.1], [.2]]), np.repeat([0, 2], 16)),
])
def test_invalid_saved_grid_is_rejected(scores, gt):
    with pytest.raises(ValueError):
        score_diagnostics(scores, gt)


def test_original_logits_preserve_ranking_lost_to_probability_rounding():
    logits = torch.tensor([[0., 25.], [0., 30.], [25., 0.], [30., 0.]])
    rounded = 1 - logits.softmax(-1)[:, 0]
    assert rounded[0] == rounded[1] == 1
    assert rounded[2] == rounded[3] == 0
    margin = alignment_log_odds(logits)
    torch.testing.assert_close(margin, torch.tensor([25., 30., -25., -30.]))
    reference = (1 - logits.double().softmax(-1)[:, 0])
    assert torch.equal(torch.argsort(margin), torch.argsort(reference))


def test_log_odds_matches_softmax_group_probability_with_multiple_classes():
    logits = torch.tensor([[.3, -.2, .4], [-.1, .2, -.7]], dtype=torch.float64)
    margin = alignment_log_odds(logits, normal_index=1)
    torch.testing.assert_close(margin.sigmoid(), 1 - logits.softmax(-1)[:, 1])


def test_split_report_keeps_zero_classes_and_blocks_incomplete_validation(tmp_path):
    rows = [{"split": "train", "group": "one", "label": "B4-B5-B1"},
            {"split": "development", "group": "two", "label": "A"}]
    report = split_coverage(rows, "xd")
    assert report["train"]["label_counts"]["B5"] == 1
    assert report["development"]["missing_labels"] == ["B1", "B2", "B4", "B5", "B6", "G"]
    labels = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    with pytest.raises(ValueError, match="No training started"):
        validate_split_coverage(labels, np.array([0, 1, 2]), np.array([3]),
                                ["normal", "shooting", "abuse"], tmp_path / "split_health.json")
    assert (tmp_path / "split_health.json").is_file()
