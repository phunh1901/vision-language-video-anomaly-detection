"""Crop identity, native exposure, pooling, ranking ties and no-GPU failure gates."""

import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from test_v21_native import notebook_scope
from v21_data import dataset_split
from v21_error_analysis import ranking_tail
from v21_recipe import FullCropPack, discover_train_root, recipe_batches, repository_video_input


def crop_fixture(tmp_path, *, zip_input=False, crops=10):
    rows = [{"path": f"original/{name}__{crop}.npy", "label": label}
            for name, label in (("Normal001", "Normal"), ("Abuse001", "Abuse"))
            for crop in range(10)]
    root = tmp_path / ("features.zip" if zip_input else "features")
    if not zip_input:
        root.mkdir()
    archive = ZipFile(root, "w") if zip_input else None
    for i, row in enumerate(rows):
        crop = i % 10
        if crop >= crops:
            continue
        name = Path(row["path"]).name
        buffer = io.BytesIO()
        np.save(buffer, np.full((4, 512), i, dtype=np.float16))
        if archive:
            archive.writestr("original/" + name, buffer.getvalue())
        else:
            (root / name).write_bytes(buffer.getvalue())
    if archive:
        archive.close()
    return root, rows


@pytest.mark.parametrize("zip_input", [False, True])
def test_all_crop_identities_survive_loading_and_split(tmp_path, zip_input):
    root, rows = crop_fixture(tmp_path, zip_input=zip_input)
    pack = FullCropPack(root, "ucf", rows)
    try:
        pack.validate(tmp_path / "audit")
        assert len(pack.features) == len(pack.checked) == 20
        def encode(labels):
            return torch.tensor([[x == "Normal", x == "Abuse"] for x in labels])
        ids, groups, _, _, train, dev, _, _ = dataset_split(
            rows, [{"path": "other__0.npy"}], pack.features, encode, "ucf", 234,
            "official_train_final_epoch", preserve_crops=True)
        assert len(set(ids)) == 20 and len(set(groups)) == 2
        assert len(train) == 20 and not len(dev)
        assert not np.array_equal(pack.array(pack.features[ids[0]]), pack.array(pack.features[ids[1]]))
    finally:
        pack.close()


def test_one_crop_is_rejected_instead_of_replicated(tmp_path):
    root, rows = crop_fixture(tmp_path, crops=1)
    with pytest.raises(ValueError, match="missing 18"):
        FullCropPack(root, "ucf", rows)


def test_preserving_crops_cannot_hide_official_train_test_video_overlap(tmp_path):
    root, rows = crop_fixture(tmp_path)
    pack = FullCropPack(root, "ucf", rows)
    with pytest.raises(ValueError, match="official train"):
        dataset_split(rows, [{"path": "Normal001__9.npy"}], pack.features,
                      lambda labels: None, "ucf", 234, preserve_crops=True)


def test_nonfinite_original_crop_is_identified_before_training(tmp_path):
    root, rows = crop_fixture(tmp_path)
    np.save(root / "Abuse001__7.npy", np.full((4, 512), np.nan))
    pack = FullCropPack(root, "ucf", rows)
    with pytest.raises(ValueError, match="Abuse001__7.npy"):
        pack.validate(tmp_path / "audit")


@pytest.mark.parametrize("zip_input", [False, True])
def test_none_train_root_discovers_complete_attached_features(tmp_path, zip_input):
    root, rows = crop_fixture(tmp_path, zip_input=zip_input)
    resolved = discover_train_root(None, tmp_path, "ucf", rows, tmp_path / "audit")
    assert resolved == (root if zip_input else tmp_path)
    pack = FullCropPack(resolved, "ucf", rows)
    assert len(pack.features) == 20
    pack.close()
    receipt = json.loads((tmp_path / "audit/input_discovery.json").read_text())
    assert not receipt["gpu_work_started"] and receipt["required_crop_features"] == 20


def test_none_root_reports_legacy_pack_inventory_instead_of_unset_variable(tmp_path):
    rows = [{"path": f"normal__{i}.npy", "label": "Normal"} for i in range(10)]
    with ZipFile(tmp_path / "train.zip", "w") as archive:
        archive.writestr("vadclip_ucf_train_v5_manifest.json", json.dumps({
            "schema": "vadclip_v5_train_v1", "dataset": "ucf", "crop": 5, "features": [{}]}))
    with pytest.raises(ValueError, match="Found one-crop v5 pack"):
        discover_train_root(None, tmp_path, "ucf", rows, tmp_path / "audit")
    receipt = json.loads((tmp_path / "audit/input_discovery.json").read_text())
    assert receipt["inputs_found"][0]["crops_per_video"] == 1
    assert receipt["inputs_found"][0]["unique_videos"] == 1


def test_discovery_does_not_arbitrarily_choose_duplicate_complete_inputs(tmp_path):
    root, rows = crop_fixture(tmp_path, zip_input=True)
    (tmp_path / "another.zip").write_bytes(root.read_bytes())
    with pytest.raises(ValueError, match="Multiple complete ucf inputs"):
        discover_train_root(None, tmp_path, "ucf", rows, tmp_path / "audit")


def test_inventory_detects_changed_feature_after_preflight(tmp_path):
    root, rows = crop_fixture(tmp_path)
    pack = FullCropPack(root, "ucf", rows)
    pack.validate(tmp_path / "audit")
    new = FullCropPack(root, "ucf", rows)
    new.load_inventory(tmp_path / "audit/validated_crop_inventory.json")
    item = next(iter(new.features.values()))
    np.save(root / item["path"], np.zeros((4, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="changed after validation"):
        new.array(item)


@pytest.mark.parametrize("dataset,normal,abnormal,size,updates,exposure", [
    ("ucf", 8000, 8100, 128, 125, 16000),
    ("xd", 20490, 19050, 96, 412, 39540),
])
def test_native_epoch_exposure_matches_original_loader_semantics(dataset, normal, abnormal, size, updates, exposure):
    labels = np.r_[np.zeros(normal, bool), np.ones(abnormal, bool)]
    config = {"native_repository_recipe": True, "dataset": dataset, "batch_size": size}
    batches = recipe_batches(np.arange(len(labels)), labels, config, 234)
    assert len(batches) == updates and sum(map(len, batches)) == exposure
    assert len(set(np.concatenate(batches))) == exposure
    if dataset == "ucf":
        assert all(labels[b].sum() == 64 for b in batches)
    else:
        assert len(batches[-1]) == 84  # native XD does not drop the final batch


@pytest.mark.parametrize("steps", [3, 8, 27])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_temporal_means_match_captured_upstream_process_feat(steps, dtype):
    # Functions taken from the pinned source; no network or training needed.
    source = (Path(__file__).resolve().parents[1]
              / "versions/_dev/v21/fixtures/pinned_tools.txt").read_text(encoding="utf-8")
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name in {"random_extract", "uniform_extract", "pad", "process_feat"}]
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned_tools", "exec"), namespace)
    array = np.random.default_rng(3).normal(size=(steps, 512)).astype(dtype)
    expected, expected_length = namespace["process_feat"](array, 8)
    trace = {}
    actual, lengths, n = repository_video_input(array, 8, "cpu", rng=True, trace=trace)
    np.testing.assert_array_equal(actual.numpy()[0], expected)
    assert n == expected_length == int(lengths[0])
    if steps > 8:
        assert trace["pooling_ranges"][0][0] == 0 and trace["pooling_ranges"][-1][1] == steps


def test_top_fraction_ties_are_order_invariant_and_fractional():
    y, score = np.array([1, 0, 0, 1]), np.array([.9, .9, .9, .1])
    info, weights = ranking_tail(y, score, .25)
    assert info["precision"] == pytest.approx(1/3)
    assert weights.sum() == pytest.approx(1)
    assert info["false_positive_frame_weight"] == pytest.approx(2/3)
    other, _ = ranking_tail(y[::-1], score[::-1], .25)
    assert other == info


def test_failure_in_either_cpu_input_gate_prevents_all_gpu_jobs(tmp_path):
    scope = notebook_scope()
    scope.update(INPUT=tmp_path, WORK=tmp_path)
    scope["load_module"] = lambda *args: SimpleNamespace(
        collect_reports=lambda *args: None, collect_latency=lambda *args: None)
    scope["resolve_checkpoint"] = lambda *args: pytest.fail("GPU evaluation path must not start")
    calls = []

    def run(name, source, cfg, scripts):
        assert cfg["CPU_PREFLIGHT"]
        calls.append(name)
        if name.startswith("xd"):
            raise ValueError("missing 9 crop views")
        return tmp_path / "preflight.json"

    scope["run_dataset"] = run
    with pytest.raises(RuntimeError, match="CPU crop validation failed"):
        scope["run_both"]()
    assert calls == ["ucf_crop_preflight", "xd_crop_preflight"]
    with ZipFile(scope["LAST_RESULT_ZIP"]) as archive:
        report = json.loads(archive.read("summary.json"))
    assert not report["cosine_control_trained"]


def test_native_full_routes_preflight_both_then_baseline_train_and_evaluation(tmp_path):
    scope = notebook_scope()
    scope.update(INPUT=tmp_path, WORK=tmp_path)
    scope["resolve_checkpoint"] = lambda *args: tmp_path / "v0.pth"
    scope["load_module"] = lambda *args: SimpleNamespace(
        collect_reports=lambda *args: None, collect_latency=lambda *args: None,
        analyze_benchmark=lambda *args: {"error_segment_count": 0})
    scope["verify_comparison"] = lambda *args: []
    calls = []

    def run(name, source, cfg, scripts):
        calls.append(name)
        ds = name.split("_")[0]
        if cfg.get("CPU_PREFLIGHT"):
            return tmp_path / "preflight.json"
        training = name.endswith("_train")
        path = cfg["OUTPUT"] / ("training.json" if training else "run.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if training:
            assert cfg["TRAIN_CONFIG"]["epochs"] == 10
            assert cfg["TRAIN_CONFIG"]["batch_size"] == (128 if ds == "ucf" else 96)
            payload = {"variants": {"cosine_control": {"checkpoint_file": "cosine_control/best.pth",
                       "space": "native_cosine", "checkpoint_sha256": "fixture"}}}
        else:
            branch, metric, value = (("classification", "roc_auc", .8801992605166816)
                                     if ds == "ucf" else ("branch_2", "frame_ap", .8450448809004643))
            payload = {"metrics": {branch: {metric: value}}}
        path.write_text(json.dumps(payload))
        return path

    scope["run_dataset"] = run
    result = scope["run_both"]()
    assert calls == ["ucf_crop_preflight", "xd_crop_preflight", "ucf_v0", "ucf_train",
                     "ucf_cosine_control", "xd_v0", "xd_train", "xd_cosine_control"]
    with ZipFile(result) as archive:
        report = json.loads(archive.read("summary.json"))
        assert "historical_comparison.csv" not in archive.namelist()
    assert report["cosine_control_trained"] and report["training_recipe"] == "COSINE_PINNED_REPOSITORY"
