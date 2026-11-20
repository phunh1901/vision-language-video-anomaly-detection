"""Build standalone V21 from the existing V0 evaluator without editing older versions."""

import json
import sys
from pathlib import Path

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"



HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Missing/ambiguous pinned runner marker: {old!r}")
    return source.replace(old, new)


def build(recipe="A"):
    if recipe not in {"A", "B", "PAPER", "COSINE"}:
        raise ValueError("Recipe must be A, B, PAPER or COSINE")
    original = json.loads((NOTEBOOKS / "V00.ipynb").read_text(encoding="utf-8"))
    base = {}
    for index in (3, 5, 6):
        exec("".join(original["cells"][index]["source"]), base)
    previous = json.loads((NOTEBOOKS / "V20.ipynb").read_text(encoding="utf-8"))
    v20 = {}
    exec("".join(previous["cells"][2]["source"]), v20)
    sources = {
        name: (HERE / name).read_text(encoding="utf-8")
        for name in ("v21_geometry.py", "v21_model.py", "v21_native.py", "v21_data.py", "v21_schedule.py", "v21_optimizer.py", "v21_recipe.py", "v21_flow.py", "v21_health.py", "v21_diagnostics.py", "v21_error_analysis.py", "v21_recovery.py", "train_v21.py")
    }
    sources.update({
        "latency_inference.py": (SHARED / "latency_inference.py").read_text(encoding="utf-8"),
        "benchmark_reporting.py": (SHARED / "benchmark_reporting.py").read_text(encoding="utf-8"),
        "analysis.py": v20["SOURCES"]["analysis.py"],
        "v7_windows.py": base["WINDOW_SOURCE"],
        "orchestrator_v21.py": (HERE / "run_v21.py").read_text(encoding="utf-8"),
    })
    ucf = once(base["UCF_RUNNER"], 'model.to(device).eval()\n', '''model.to(device).eval()
from v21_model import load_full_model
from v21_native import load_native_model

full_model, full_provenance = None, None
if V21_SPACE == "native_cosine":
    full_model, full_provenance = load_native_model(
        model, V21_MODEL_PATH, V21_MODEL_SHA256, evaluated_sha256, device)
    model.forward = full_model.forward
elif V21_SPACE is not None:
    full_model, full_provenance = load_full_model(
        model, V21_SPACE, GEOMETRY_CONFIG, V21_MODEL_PATH, V21_MODEL_SHA256,
        evaluated_sha256, device)
    model.forward = full_model.forward
''')
    ucf = once(ucf, '"parameters": count_parameters(model, None),',
               '"parameters": count_parameters(model, (full_model.geometry,) if hasattr(full_model, "geometry") else None),\n'
               '    "geometry": full_model.geometry.receipt() if hasattr(full_model, "geometry") else None,\n'
               '    "model_control": full_model.model_receipt() if V21_SPACE == "native_cosine" else None,\n'
               '    "full_model_provenance": full_provenance,')
    ucf = once(ucf, '"evaluated_checkpoint_sha256": evaluated_sha256,',
               '"initial_v0_checkpoint_sha256": evaluated_sha256,\n'
               '    "evaluated_checkpoint_sha256": V21_MODEL_SHA256 if full_model else evaluated_sha256,')
    xd = once(base["XD_RUNNER"], '        original_forward = CLIPVAD.forward\n',
              '''        from v21_model import load_full_model
        from v21_native import load_native_model

        full_model, full_provenance = None, None
        original_forward = CLIPVAD.forward
''')
    xd = once(xd, '            nonlocal forwards\n',
              '            nonlocal forwards, full_model, full_provenance\n')
    xd = once(xd, '                parameter_counts.update(count_parameters(self, None))',
               '''                if V21_SPACE == "native_cosine":
                    full_model, full_provenance = load_native_model(
                        self, V21_MODEL_PATH, V21_MODEL_SHA256, checkpoint_hash,
                        images.device, native_forward=original_forward)
                elif V21_SPACE is not None:
                    full_model, full_provenance = load_full_model(
                        self, V21_SPACE, GEOMETRY_CONFIG, V21_MODEL_PATH, V21_MODEL_SHA256,
                        checkpoint_hash, images.device)
                parameter_counts.update(count_parameters(
                    self, (full_model.geometry,) if hasattr(full_model, "geometry") else None))''')
    xd = once(xd, '                return original_forward(self, x, mask, prompts, sizes)',
              '''                if full_model is not None:
                    return full_model(x, mask, prompts, sizes)
                return original_forward(self, x, mask, prompts, sizes)''')
    xd = once(xd, '        **receipt,', '''        **receipt,
        "geometry": full_model.geometry.receipt() if hasattr(full_model, "geometry") else None,
        "model_control": full_model.model_receipt() if V21_SPACE == "native_cosine" else None,
        "full_model_provenance": full_provenance,
        "initial_v0_checkpoint_sha256": checkpoint_hash,
        "evaluated_checkpoint_sha256": V21_MODEL_SHA256 if full_model else checkpoint_hash,''')
    # Diagnostic forward is separate from latency.measure, and never receives frame GT.
    ucf = once(ucf, 'latency = InferenceLatency(LATENCY_WARMUP, LATENCY_REPEATS)',
               '''latency = InferenceLatency(LATENCY_WARMUP, LATENCY_REPEATS)
from v21_diagnostics import EvaluationTrace
flow_trace = EvaluationTrace(OUTPUT / ("smoke" if SMOKE_ONLY else "full") / "diagnostics", "ucf", VARIANT)''')
    ucf = once(ucf, 'def predict_video(feature, model, device, block_length):',
               'def predict_video(feature, model, device, block_length, video_id):')
    ucf = once(ucf, '        _, logits_c, logits_a = latency.measure(',
               '        measured = latency.measure(')
    ucf = once(ucf, '        c_score = logits_c.reshape(-1)[:steps].sigmoid()',
               '''        _, logits_c, logits_a = measured
        flow_trace.observe(video_id, model, dispatch, visual, lengths, mask, measured, steps,
                           getattr(full_model, "geometry", None), list(CLASSES), expected_features=feature)
        c_score = logits_c.reshape(-1)[:steps].sigmoid()''')
    ucf = once(ucf, 'scores = predict_video(feature, model, device, args.visual_length)',
               'scores = predict_video(feature, model, device, args.visual_length, row["id"])')
    ucf = once(ucf, 'selected = sorted(scores_by_index)', 'flow_trace.finish()\nselected = sorted(scores_by_index)')
    xd = once(xd, '        latency = InferenceLatency(LATENCY_WARMUP, LATENCY_REPEATS)',
              '''        latency = InferenceLatency(LATENCY_WARMUP, LATENCY_REPEATS)
        from v21_diagnostics import EvaluationTrace
        flow_trace = EvaluationTrace(output_dir / "diagnostics", "xd", VARIANT)
        with resolved_csv.open(newline="", encoding="utf-8-sig") as feature_stream:
            feature_paths = [Path(item["path"]) for item in csv.DictReader(feature_stream)]
        if len(feature_paths) != len(inventory):
            raise ValueError("XD resolved feature paths do not match evaluator inventory")''')
    xd = once(xd, '            forwards += 1', '''            flow_trace.observe(row["official_path"], self, dispatch, images, lengths,
                               padding_mask, result, row["steps"], getattr(full_model, "geometry", None), text,
                               expected_features=np.load(feature_paths[forwards], allow_pickle=False))
            forwards += 1''')
    xd = once(xd, '    latency_summary = latency.save(', '    flow_trace.finish()\n    latency_summary = latency.save(')
    suffix = "" if recipe == "A" else f"_{recipe}"
    config = (HERE / f"notebook_config{suffix}.py").read_text(encoding="utf-8")
    introduction = (HERE / f"NOTEBOOK_GUIDE{suffix}.md").read_text(encoding="utf-8")
    payload = ('SOURCES = ' + repr(sources) + '\n\nUCF_RUNNER = ' + repr(ucf)
               + '\n\nXD_RUNNER = ' + repr(xd)
               + '\n\nTRAINER_SOURCE = SOURCES["train_v21.py"]\n')
    if recipe == "COSINE":
        historical = json.loads((HERE / "v21_zip5_reference.json").read_text(encoding="utf-8"))
        payload += "\nZIP5_REFERENCE = " + repr(historical) + "\n"

    def cell(kind, source):
        item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        return item

    download = '''import shutil
from pathlib import Path
from IPython.display import FileLink, display

result_zip = globals().get("RESULT_ZIP", globals().get("LAST_RESULT_ZIP"))
if result_zip is None:
    raise RuntimeError("No result ZIP yet. Check the run cell error and output directory.")
source_zip = Path(result_zip).resolve()
if not source_zip.is_file():
    raise FileNotFoundError(source_zip)
download_zip = Path.cwd() / f"v21_{source_zip.parent.name}_results.zip"
if source_zip != download_zip.resolve():
    shutil.copy2(source_zip, download_zip)
display(FileLink(download_zip.name))
print("If the link fails, use Kaggle Output/Files:", download_zip)
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": original["metadata"], "cells": [
        cell("markdown", introduction), cell("code", config), cell("code", payload),
        cell("code", (HERE / "run_v21.py").read_text(encoding="utf-8")),
        cell("code", download),
    ]}
    path = (NOTEBOOKS / "V21.ipynb" if recipe == "COSINE" else
            HERE / "fixtures" / "recipe_notebooks" / f"run_vadclip_v21{suffix or '_A'}.ipynb.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path.name)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "COSINE")
