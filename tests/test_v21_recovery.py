"""Reuse postprocessing failures without retraining or silently mixing recipes."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from test_v21_native import notebook_scope
from v21_recovery import RecoveryInput


def saved_case(root):
    root.mkdir()
    sources = {"train_v21.py": "trainer", "v21_model.py": "model", "orchestrator_v21.py": "old audit"}
    cfg = {"epochs": 30, "learning_rate": 1e-5}
    experiment = {"mode": "full", "seed": 42, "geometry_config": {},
                  "source_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}}
    (root / "experiment.json").write_text(json.dumps(experiment))
    weight = root / "ucf/train/cosine_control/best.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fixture-checkpoint")
    sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    record = {"config": cfg, "initial_v0_sha256": "v0", "checkpoint_file": "cosine_control/best.pth",
              "checkpoint_sha256": sha}
    (root / "ucf/train/training.json").write_text(json.dumps({
        "initial_v0_sha256": "v0", "feature_manifest_sha256": "manifest", "variants": {"cosine_control": record}}))
    for name in ("v0", "cosine_control"):
        folder = root / f"ucf/{name}/full"
        (folder / "diagnostics").mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"initial_v0_checkpoint_sha256": "v0",
            "evaluated_checkpoint_sha256": "v0" if name == "v0" else sha,
            "run_kind": "full_checkpoint_inference", "metrics": {"fixture": True}}))
        (folder / "predictions.npz").write_bytes(b"fixture-predictions")
        (folder / "diagnostics/diagnostic_tensors.npz").write_bytes(b"fixture-raw-logits")
    (root / "ucf/error_analysis").mkdir()
    (root / "ucf/error_analysis/incomplete.csv").write_text("old incomplete analysis")
    return sources, cfg


@pytest.mark.parametrize("as_zip", [False, True])
def test_recovery_copies_validated_complete_dataset_only_and_retains_provenance(tmp_path, as_zip):
    source = tmp_path / "old"
    sources, cfg = saved_case(source)
    origin = source
    if as_zip:
        origin = tmp_path / "previous_results.zip"
        with ZipFile(origin, "w") as archive:
            for path in source.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(source))
    output = tmp_path / "new"
    with RecoveryInput(origin) as old:
        assert old.reuse("xd", output, sources, cfg, {}, 42, "v0", "manifest", "cosine_control") is None
        paths, training = old.reuse("ucf", output, sources, cfg, {}, 42, "v0", "manifest", "cosine_control")
    assert set(paths) == {"v0", "cosine_control"} and training.is_file()
    assert not (output / "ucf/error_analysis").exists()
    receipt = json.loads((output / "ucf/recovery_receipt.json").read_text())
    assert not receipt["new_training"] and not receipt["new_inference"]
    assert receipt["source_experiment"]["seed"] == 42


@pytest.mark.parametrize("bad", ["mode", "seed", "source", "lr", "checkpoint", "manifest", "missing_prediction"])
def test_recovery_rejects_wrong_recipe_inputs_or_checkpoint(tmp_path, bad):
    source = tmp_path / "old"
    sources, cfg = saved_case(source)
    seed, manifest = 42, "manifest"
    if bad == "mode":
        p = source / "experiment.json"
        payload = json.loads(p.read_text())
        payload["mode"] = "pilot"
        p.write_text(json.dumps(payload))
    elif bad == "seed":
        seed = 7
    elif bad == "source":
        sources["train_v21.py"] += "modified"
    elif bad == "lr":
        cfg = {**cfg, "learning_rate": 5e-4}
    elif bad == "checkpoint":
        (source / "ucf/train/cosine_control/best.pth").write_bytes(b"corrupted")
    elif bad == "manifest":
        manifest = "another-manifest"
    else:
        (source / "ucf/v0/full/predictions.npz").unlink()
    with RecoveryInput(source) as old, pytest.raises((ValueError, FileNotFoundError)):
        old.reuse("ucf", tmp_path / "new", sources, cfg, {}, seed, "v0", manifest, "cosine_control")


@pytest.mark.parametrize("changed_training", [False, True])
def test_recovery_accepts_inference_trace_fix_but_rejects_training_diagnostic_change(tmp_path, changed_training):
    source = tmp_path / "old"
    sources, cfg = saved_case(source)
    old = "def check_batch(x):\n    return x\n\nclass EvaluationTrace:\n    contract = 'old'\n"
    new = old.replace("'old'", "'new'")
    if changed_training:
        new = new.replace("return x", "return x+1")
    scripts = source / "runtime_scripts"
    scripts.mkdir()
    (scripts / "v21_diagnostics.py").write_bytes(old.encode())
    metadata_path = source / "experiment.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"]["v21_diagnostics.py"] = hashlib.sha256(old.encode()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    sources["v21_diagnostics.py"] = new
    with RecoveryInput(source) as saved:
        if changed_training:
            with pytest.raises(ValueError, match="changed training behavior"):
                saved.reuse("ucf", tmp_path / "new", sources, cfg, {}, 42, "v0", "manifest", "cosine_control")
        else:
            paths, _ = saved.reuse("ucf", tmp_path / "new", sources, cfg, {}, 42, "v0", "manifest", "cosine_control")
            assert set(paths) == {"v0", "cosine_control"}


def test_run_full_reuses_ucf_and_runs_xd_without_ucf_train_or_inference(tmp_path):
    scope = notebook_scope(legacy=True)
    scope.update(INPUT=tmp_path, WORK=tmp_path, RUN_MODE="full", RECOVERY_FROM="old-run")
    scope["resolve_checkpoint"] = lambda *args: tmp_path / "v0.pth"
    real_load = scope["load_module"]
    calls = []

    class RecoveryFixture:
        def __init__(self, source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def reuse(self, ds, output, sources, cfg, *args):
            if ds == "xd":
                return None
            paths = {}
            for variant in ("v0", "cosine_control"):
                paths[variant] = create_report(ds, variant, Path(output) / ds / variant)
            return paths, create_training(ds, Path(output) / ds / "train")

    def create_report(ds, variant, output):
        output.mkdir(parents=True, exist_ok=True)
        reference = scope["ZIP5_REFERENCE"]["datasets"][ds]
        payload = {**reference["protocol"], "metrics": copy.deepcopy(reference["metrics"])}
        if variant == "v0":
            branch, key, value = (("classification", "roc_auc", .8801992605166816) if ds == "ucf" else
                                  ("branch_2", "frame_ap", .8450448809004643))
            payload["metrics"][branch][key] = value
        path = output / "run.json"
        path.write_text(json.dumps(payload))
        return path

    def create_training(ds, output):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "training.json"
        path.write_text(json.dumps({**scope["ZIP5_REFERENCE"]["datasets"][ds]["training"],
            "variants": {"cosine_control": {"space": "native_cosine", "checkpoint_file": "best.pth",
                                               "checkpoint_sha256": "fixture"}}}))
        return path

    def run(name, source, cfg, scripts):
        calls.append(name)
        ds, kind = name.split("_", 1)
        return create_training(ds, cfg["OUTPUT"]) if kind == "train" else create_report(ds, kind, cfg["OUTPUT"])

    def load(name, path):
        if name == "v21_recovery":
            return SimpleNamespace(RecoveryInput=RecoveryFixture)
        if name == "v21_error_analysis":
            return SimpleNamespace(analyze_benchmark=lambda *args: {"error_segment_count": 0})
        return real_load(name, path)

    scope.update(run_dataset=run, load_module=load, verify_comparison=lambda *args: [])
    # Only orchestration routing is under test; metric collectors have separate tests.
    reporting = SimpleNamespace(collect_reports=lambda *args: None, render_table=lambda *args, **kwargs: "table")
    collector_load = scope["load_module"]
    scope["load_module"] = lambda name, path: (reporting if name == "v21_reporting" else
        SimpleNamespace(collect_latency=lambda *args: None) if name == "v21_latency" else collector_load(name, path))
    result = scope["run_both"]()
    assert calls == ["xd_v0", "xd_train", "xd_cosine_control"]
    with ZipFile(result) as archive:
        summary = json.loads(archive.read("summary.json"))
    assert summary["runs"]["ucf"]["reused_completed_dataset"]
    assert all(row["status"] == "ok" for row in summary["runs"].values())
