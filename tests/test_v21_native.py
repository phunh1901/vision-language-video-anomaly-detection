"""Native-control behavior using CPU fixtures; these are not upstream/GPU benchmarks."""

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from test_v21_flow import FlowBase
from train_v21 import fit
from v21_flow import EncoderFlowAudit
from v21_native import NATIVE_SCHEMA, NativePipeline, load_native_model, native_objective

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"


HERE = (REPOSITORY / 'versions/_dev/v21')

# A deliberately small source fixture exercising actual extraction/data flow.
# It is not represented as the source of the author's CLAS/CLAS2 functions.
LOSS_SOURCE = '''
def CLAS(logits, labels, lengths, device):
    pooled = torch.stack([logits[i, :int(n)].topk(1 + int(n)//16, dim=0).values.mean(0)
                          for i, n in enumerate(lengths)])
    labels = labels / labels.sum(-1, keepdim=True)
    return -(labels * pooled.log_softmax(-1)).sum(-1).mean()

def CLAS2(logits, labels, lengths, device):
    scores = torch.stack([logits[i, :int(n), 0].sigmoid().max()
                          for i, n in enumerate(lengths)])
    return F.binary_cross_entropy(scores, 1 - labels[:, 0])

def train(model, visual, padding_mask, prompt_text, lengths, labels, device):
    for epoch in range(2):
        text_features, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
        loss1 = CLAS2(logits1, labels, lengths, device)
        loss_total1 += loss1.item()
        loss2 = CLAS(logits2, labels, lengths, device)
        loss_total2 += loss2.item()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        loss3 = torch.zeros(1).to(device)
        text_feature_normal = text_features[0]
        for i in range(1, text_features.shape[0]):
            text_feature_abnormal = text_features[i]
            loss3 += torch.abs(text_feature_normal @ text_feature_abnormal)
        loss3 = loss3 / (text_features.shape[0] - 1)
        loss = loss1 + loss2 + loss3 * 0.1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''


class NativeFixture(FlowBase):
    """Explicit native-style heads/fusion fixture; no Poincare mapping."""

    def encode_video(self, images, padding_mask, lengths):
        return super().encode_video(images[..., :8], padding_mask, lengths)

    def forward(self, images, padding_mask, text, lengths):
        visual = self.encode_video(images, padding_mask, lengths)
        c = self.classifier(visual + self.mlp2(visual))
        raw = self.encode_textprompt(text)
        prompt = torch.nn.functional.normalize((c.sigmoid() * visual).sum(1), dim=-1)
        fused = raw[None] + self.mlp1(prompt[:, None] + raw[None])
        a = (torch.nn.functional.normalize(visual, dim=-1)
             @ torch.nn.functional.normalize(fused, dim=-1).transpose(-1, -2)) / .07
        return raw, c, a


def config():
    scope = {}
    exec((HERE / "notebook_config_COSINE.py").read_text(encoding="utf-8"), scope)
    return copy.deepcopy(scope["TRAIN_CONFIG"]), scope["GEOMETRY_CONFIG"]


def fixture_inputs():
    torch.manual_seed(7)
    return (torch.randn(2, 6, 512), torch.tensor([6, 3]),
            torch.tensor([[1., 0., 0.], [0., 1., 0.]]))


def test_extracted_objective_matches_independent_reference_and_masks_padding():
    objective, receipt = native_objective(LOSS_SOURCE)
    torch.manual_seed(1)
    raw = torch.randn(3, 8, requires_grad=True)
    c, a = torch.randn(2, 6, 1, requires_grad=True), torch.randn(2, 6, 3, requires_grad=True)
    targets = torch.tensor([[1., 0., 0.], [0., .5, .5]])
    sizes = torch.tensor([6, 3])
    loss, parts = objective(c, a, targets, sizes, {"encoder_text": raw}, "native_cosine", {})
    bce = torch.nn.functional.binary_cross_entropy(
        torch.stack((c[0, :6].sigmoid().max(), c[1, :3].sigmoid().max())),
        torch.tensor([0., 1.]))
    pooled = torch.stack((a[0, :6].max(0).values, a[1, :3].max(0).values))
    ce = -(targets * pooled.log_softmax(-1)).sum(-1).mean()
    unit = raw / raw.norm(dim=-1, keepdim=True)
    contrastive = (unit[1:] @ unit[0]).abs().mean()
    torch.testing.assert_close(loss[0], bce + ce + .1 * contrastive)
    torch.testing.assert_close(parts["contrastive"][0], contrastive)
    assert "loss_total" not in receipt["extracted_source"]
    assert "optimizer" not in receipt["extracted_source"]
    loss.mean().backward()
    assert not a.grad[1, 3:].any() and not c.grad[1, 3:].any()
    assert raw.grad.norm() > 0
    objective_xd, _ = native_objective(LOSS_SOURCE.replace("loss3 * 0.1", "loss3 * 0.0001"))
    xd_loss, _ = objective_xd(c, a, targets, sizes, {"encoder_text": raw}, "native_cosine", {})
    torch.testing.assert_close(xd_loss[0], bce + ce + .0001 * contrastive)


def native_symbol_fixture(coefficient=.1):
    """Reproduce the CLASM/feat_lengths/text_labels symbols from the Kaggle error.

    This transforms our test fixture; it is not a copy of the upstream trainer.
    """
    tree = ast.parse(LOSS_SOURCE)

    class HelperNames(ast.NodeTransformer):
        mapping = {"CLAS": "CLAS2", "CLAS2": "CLASM"}

        def visit_FunctionDef(self, node):
            node.name = self.mapping.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_Name(self, node):
            node.id = self.mapping.get(node.id, node.id)
            return node

    class InputNames(ast.NodeTransformer):
        mapping = {"lengths": "feat_lengths", "labels": "text_labels"}

        def visit_Name(self, node):
            node.id = self.mapping.get(node.id, node.id)
            return node

        def visit_arg(self, node):
            node.arg = self.mapping.get(node.arg, node.arg)
            return node

    tree = HelperNames().visit(tree)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    InputNames().visit(train)
    return ast.unparse(ast.fix_missing_locations(tree)).replace(
        "loss3 * 0.1", f"loss3 * {coefficient}")


@pytest.mark.parametrize("coefficient", [.1, .0001])
def test_actual_error_symbols_bind_to_encoded_targets_and_valid_lengths(coefficient):
    source = native_symbol_fixture(coefficient)
    objective, receipt = native_objective(source, "symbol_regression.py")
    reference, _ = native_objective(LOSS_SOURCE.replace("loss3 * 0.1", f"loss3 * {coefficient}"))
    assert receipt["input_bindings"]["text_labels"] == "_native_targets"
    assert receipt["input_bindings"]["feat_lengths"] == "_native_lengths"
    assert receipt["native_loss_helpers"] == ["CLAS2", "CLASM"]
    assert receipt["extraction_contract"] == 2
    torch.manual_seed(12)
    args = (torch.randn(2, 6, 1, requires_grad=True), torch.randn(2, 6, 3, requires_grad=True),
            torch.tensor([[1., 0., 0.], [0., .5, .5]]), torch.tensor([6, 3]),
            {"encoder_text": torch.randn(3, 8, requires_grad=True)}, "native_cosine", {})
    actual, actual_parts = objective(*args)
    expected, expected_parts = reference(*args)
    torch.testing.assert_close(actual, expected)
    for key in actual_parts:
        torch.testing.assert_close(actual_parts[key], expected_parts[key])
    actual_grads = torch.autograd.grad(actual.sum(), (args[0], args[1], args[4]["encoder_text"]))
    expected_grads = torch.autograd.grad(expected.sum(), (args[0], args[1], args[4]["encoder_text"]))
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b)
    assert not actual_grads[0][1, 3:].any() and not actual_grads[1][1, 3:].any()


def test_extraction_resolves_helpers_called_by_native_loss_helpers():
    source = native_symbol_fixture().replace(
        "return F.binary_cross_entropy(scores, 1 - labels[:, 0])",
        "return binary_helper(scores, 1 - labels[:, 0])")
    source = "def binary_helper(scores, target):\n    return F.binary_cross_entropy(scores, target)\n\n" + source
    _, receipt = native_objective(source)
    assert "binary_helper" in receipt["native_loss_helpers"]


def test_extraction_rejects_different_label_or_length_inputs_between_branches():
    source = native_symbol_fixture()
    with pytest.raises(ValueError, match="bind encoded labels"):
        native_objective(source.replace("CLAS2(logits2, text_labels", "CLAS2(logits2, raw_labels"))
    with pytest.raises(ValueError, match="loss input signature"):
        native_objective(source.replace("CLAS2(logits2, text_labels, feat_lengths",
                                        "CLAS2(logits2, text_labels, unrelated_lengths"))


def test_native_objective_rejects_unknown_graph_or_invalid_labels():
    with pytest.raises(ValueError, match="Unresolved"):
        native_objective(LOSS_SOURCE.replace("loss3 * 0.1", "loss3 * unknown_weight"))
    with pytest.raises(ValueError, match="unambiguous"):
        native_objective(LOSS_SOURCE.replace("optimizer.step()", "optimizer.step()\n        loss = loss * 2"))
    objective, _ = native_objective(LOSS_SOURCE)
    x, sizes, targets = fixture_inputs()
    model = NativePipeline(NativeFixture())
    _, c, a, aux = model.forward_with_aux(x, None, ["n", "a", "b"], sizes)
    with pytest.raises(ValueError, match="Unknown/conflicting"):
        objective(c, a, targets * 0, sizes, aux, "native_cosine", {})


def test_native_forward_preserves_heads_and_source_dispatch_even_when_forward_is_replaced():
    model = NativePipeline(NativeFixture()).eval()
    assert not hasattr(model, "geometry")
    assert isinstance(model.base.classifier, torch.nn.Linear)
    assert isinstance(model.base.mlp2, torch.nn.Linear)
    x, lengths, _ = fixture_inputs()
    expected = model(x, None, ["n", "a", "b"], lengths)
    model.base.forward = model.forward  # UCF evaluator dispatch; must not recurse.
    actual = model.base(x, None, ["n", "a", "b"], lengths)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)


def test_native_local_global_classifier_prompt_gradient_flow_and_frozen_clip():
    torch.set_num_threads(2)
    model = NativePipeline(NativeFixture()).train()
    objective, _ = native_objective(LOSS_SOURCE)
    x, lengths, target = fixture_inputs()
    with EncoderFlowAudit(model) as audit:
        _, c, a, aux = model.forward_with_aux(x, None, ["n", "a", "b"], lengths)
        objective(c, a, target, lengths, aux, "native_cosine", {})[0].mean().backward()
        receipt = audit.receipt()
    assert not receipt["frozen_clip_gradients"] and not receipt["image_encoder_called"]
    assert all(receipt["nonzero_finite_gradient_groups"].values())
    assert "native_classifier" in receipt["nonzero_finite_gradient_groups"]
    assert not any(module._forward_hooks for module in model.modules())


def test_native_checkpoint_roundtrip_uses_captured_forward_for_xd(tmp_path):
    model = NativePipeline(NativeFixture()).eval()
    payload = {"schema": NATIVE_SCHEMA, "space": "native_cosine", "initial_v0_sha256": "v0",
               "epoch": 30, "downstream": model.downstream_state(), "native_objective": {},
               "native_forward_sha256": hashlib.sha256(
                   inspect.getsource(NativeFixture.forward).encode()).hexdigest()}
    path = tmp_path / "best.pth"
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    base = NativeFixture()
    # Real evaluators first restore the same released V0 frozen CLIP checkpoint.
    # Downstream-only checkpoints intentionally do not contain frozen CLIP.
    base.clipmodel.load_state_dict(model.base.clipmodel.state_dict())
    original = NativeFixture.forward
    base.forward = lambda *args: (_ for _ in ()).throw(AssertionError("instrumented forward"))
    loaded, receipt = load_native_model(base, path, digest, "v0", "cpu", native_forward=original)
    x, lengths, _ = fixture_inputs()
    for a, b in zip(model(x, None, ["n", "a", "b"], lengths),
                    loaded(x, None, ["n", "a", "b"], lengths), strict=True):
        torch.testing.assert_close(a, b)
    assert receipt["schema"] == NATIVE_SCHEMA
    with pytest.raises(ValueError, match="SHA-256"):
        load_native_model(NativeFixture(), path, "wrong", "v0", "cpu")
    with pytest.raises(ValueError, match="schema/provenance"):
        load_native_model(NativeFixture(), path, digest, "different", "cpu")


@pytest.mark.parametrize("repository_recipe", [False, True])
def test_native_fit_updates_native_heads_and_saves_fixed_final_with_physical_batch64(tmp_path, repository_recipe):
    torch.set_num_threads(2)
    cfg, geom = config()
    cfg.update(epochs=2, verify_encoder_flow=True, native_repository_recipe=repository_recipe,
               batch_size=64, microbatch_size=64, dataset="ucf", scheduler_milestones=[1],
               scheduler_gamma=.1, checkpoint_epochs=[1])
    objective, receipt = native_objective(LOSS_SOURCE)

    class Pack:
        features = {str(i): i for i in range(64)}

        def array(self, i):
            return np.random.default_rng(i).normal(size=(5, 512)).astype(np.float32)

    ids = [str(i) for i in range(64)]
    target = torch.tensor([[1., 0., 0.] if i < 32 else [0., 1., 0.] for i in range(64)])
    result = fit(NativeFixture(), Pack(), ids, target, target[:, 0].numpy() == 0,
                 np.arange(64), np.array([], dtype=int), SimpleNamespace(visual_length=6),
                 ["n", "a", "b"], lambda lengths, width: torch.arange(width)[None] >= lengths[:, None],
                 False, cfg, geom, tmp_path / "cosine_control", 42, 32, torch.device("cpu"),
                 "ucf", "v0", objective, receipt)
    assert result["space"] == "native_cosine" and result["best_epoch"] == 2
    assert result["selection_branch"] == "fixed_final_epoch" and result["effective_batch"] == 64
    assert not result["preflight"]["conditioning"]
    assert result["first_gradient_groups"]["classification_head"]
    assert result["first_gradient_groups"]["prompt_tokens"]
    saved = torch.load(tmp_path / result["checkpoint_file"], weights_only=True)
    assert saved["schema"] == NATIVE_SCHEMA and saved["epoch"] == 2
    assert any(n.startswith("base.mlp2.") for n in saved["downstream"])
    assert not any(n.startswith(("geometry.", "base.clipmodel.")) for n in saved["downstream"])
    health = json.loads((tmp_path / "cosine_control/learning_health.json").read_text())
    assert health[0]["microbatches"][0]["microbatch_size"] == 64


def notebook_scope(legacy=False):
    notebook = json.loads((NOTEBOOKS / "V21.ipynb").read_text(encoding="utf-8"))
    scope = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{index}", "exec")
    for index in (1, 2):
        exec("".join(notebook["cells"][index]["source"]), scope)
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    tree.body.pop()
    exec(compile(tree, "orchestrator", "exec"), scope)
    if legacy:
        exec((HERE / "fixtures/legacy_cosine_config.py").read_text(encoding="utf-8"), scope)
    return scope


def test_canonical_notebook_uses_pinned_repository_recipe_and_embeds_current_source():
    scope = notebook_scope()
    cfg = scope["TRAIN_CONFIG"]
    assert cfg["native_repository_recipe"] and cfg["epochs"] == 10
    assert cfg["optimizer"] == "AdamW" and cfg["weight_decay"] == .01
    assert cfg["gradient_clip"] is None and scope["SEED"] == 234
    assert scope["HALF_BATCH"] == 64 and scope["DATASETS"] == ("ucf", "xd")
    assert scope["RUN_MODE"] == "full" and scope["RECOVERY_FROM"] is None
    for ds, lr, batch in (("ucf", 2e-5, 128), ("xd", 1e-5, 96)):
        resolved = scope["dataset_train_config"](cfg, ds)
        assert resolved["learning_rate"] == lr and resolved["batch_size"] == batch
    assert not cfg["reproduction_disclosure"]["exact_original_training_recipe"]
    for name, source in scope["SOURCES"].items():
        compile(source, name, "exec")
        if (HERE / name).exists():
            assert source == (HERE / name).read_text(encoding="utf-8")
    for key in ("UCF_RUNNER", "XD_RUNNER"):
        compile(scope[key], key, "exec")
        assert "load_native_model" in scope[key]
    assert "native_forward=original_forward" in scope["XD_RUNNER"]


def test_legacy_cosine_pilot_only_trains_both_datasets_and_saves_truthful_metadata(tmp_path):
    scope = notebook_scope(legacy=True)
    scope.update(INPUT=tmp_path, WORK=tmp_path, RUN_MODE="pilot")
    scope["resolve_checkpoint"] = lambda *args: tmp_path / "v0.pth"
    scope["load_module"] = lambda *args: SimpleNamespace()
    calls = []

    def run(name, source, cfg, scripts):
        assert name in {"ucf_train", "xd_train"}
        assert cfg["TRAIN_CONFIG"]["epochs"] == 3
        assert cfg["TRAIN_CONFIG"]["learning_rate"] == (1e-5 if name.startswith("ucf") else 2e-5)
        calls.append(name)
        path = cfg["OUTPUT"] / "training.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"variants": {"cosine_control": {
            "checkpoint_file": "cosine_control/best.pth", "space": "native_cosine",
            "checkpoint_sha256": "fixture"}}}))
        return path

    scope["run_dataset"] = run
    result = scope["run_both"]()
    assert calls == ["ucf_train", "xd_train"]
    with ZipFile(result) as archive:
        summary = json.loads(archive.read("summary.json"))
    assert summary["training_recipe"] == "COSINE_NATIVE_LR_CONTROL"
    assert summary["cosine_control_trained"]
    assert not summary["branch_comparisons"]
    assert all(not v["reports"] for v in summary["runs"].values())


def test_legacy_control_routes_v0_train_native_and_writes_historical_comparison(tmp_path):
    scope = notebook_scope(legacy=True)
    scope.update(INPUT=tmp_path, WORK=tmp_path, RUN_MODE="full")
    scope["resolve_checkpoint"] = lambda *args: tmp_path / "v0.pth"
    scope["load_module"] = lambda *args: SimpleNamespace(
        collect_reports=lambda *args: None, collect_latency=lambda *args: None,
        analyze_benchmark=lambda *args: {"error_segment_count": 0})
    calls = []

    def run(name, source, cfg, scripts):
        calls.append(name)
        ds = name.split("_")[0]
        path = cfg["OUTPUT"] / ("training.json" if name.endswith("_train") else "run.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("_train"):
            assert cfg["TRAIN_CONFIG"]["epochs"] == 30 and cfg["HALF_BATCH"] == 32
            payload = {**scope["ZIP5_REFERENCE"]["datasets"][ds]["training"],
                       "variants": {"cosine_control": {"space": "native_cosine",
                "checkpoint_file": "cosine_control/best.pth", "checkpoint_sha256": "fixture"}}}
        else:
            ref = scope["ZIP5_REFERENCE"]["datasets"][ds]
            payload = {**ref["protocol"], "metrics": copy.deepcopy(ref["metrics"])}
            if name.endswith("_v0"):
                assert cfg["V21_SPACE"] is None
                branch, metric, value = (("classification", "roc_auc", .8801992605166816)
                                         if ds == "ucf" else ("branch_2", "frame_ap", .8450448809004643))
                payload["metrics"][branch][metric] = value
            else:
                assert name.endswith("_cosine_control") and cfg["V21_SPACE"] == "native_cosine"
        path.write_text(json.dumps(payload))
        return path

    def paired(ds, paths, analysis, treatment):
        assert treatment == "cosine_control" and set(paths) == {"v0", "cosine_control"}
        return []

    scope["run_dataset"], scope["verify_comparison"] = run, paired
    result = scope["run_both"]()
    assert calls == [f"{ds}_{kind}" for ds in ("ucf", "xd")
                     for kind in ("v0", "train", "cosine_control")]
    with ZipFile(result) as archive:
        assert "historical_comparison.csv" in archive.namelist()
        summary = json.loads(archive.read("summary.json"))
    assert summary["cosine_control_trained"]


def test_historical_comparison_rejects_different_evaluator_grid(tmp_path):
    scope = notebook_scope()
    ref = scope["ZIP5_REFERENCE"]["datasets"]["ucf"]
    payload = {**ref["protocol"], "metrics": ref["metrics"], "frames_per_feature": 8}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="protocol mismatch"):
        scope["compare_historical"]("ucf", path, scope["ZIP5_REFERENCE"])
