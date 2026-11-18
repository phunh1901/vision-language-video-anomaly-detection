"""Build a self-contained notebook using the previously verified V7 evaluators."""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"



HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected unique source marker: {old[:100]}")
    return text.replace(old, new, 1)


def runner(cell):
    return ast.literal_eval(ast.parse("".join(cell["source"])).body[0].value)


def cell(kind, text):
    item = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build():
    base = json.loads((NOTEBOOKS / "V07.ipynb").read_text(encoding="utf-8"))
    ucf, xd = (runner(base["cells"][i]) for i in (5, 6))
    ucf = replace_once(ucf, "from v7_windows import overlap_forward", """from anchor import anchor_forward, patch_model_file
model_receipt = patch_model_file(runtime_src / 'model.py')
anchor_records = []""")
    a = ucf.index('        if VARIANT == "v7":')
    b = ucf.index('        c_score = ', a)
    ucf = ucf[:a] + '''        if VARIANT == "v0":
            _, logits_c, logits_a = model(visual, mask, list(CLASSES), lengths)
        else:
            _, logits_c, logits_a = anchor_forward(
                model, model, visual, mask, list(CLASSES), lengths,
                ANCHOR_CONFIG, anchor_records,
            )
''' + ucf[b:]
    ucf = replace_once(ucf, '    return predict_video(feature, model, device, args.visual_length)',
                      '''    result = predict_video(feature, model, device, args.visual_length)
    if VARIANT != "v0":
        anchor_records[-1]["video_id"] = row["id"]
    return result''')
    ucf = replace_once(ucf, '    "metrics": None,',
                      '    "metrics": None, "anchor_config": ANCHOR_CONFIG, **model_receipt,')
    ucf = replace_once(ucf, 'stage_directory.cleanup()', '''(output_dir / "anchor_diagnostics.json").write_text(
    json.dumps(anchor_records, indent=2), encoding="utf-8")
stage_directory.cleanup()''')
    xd = replace_once(xd, '    from v7_windows import overlap_forward', '''    from anchor import anchor_forward, patch_model_file
    model_receipt = patch_model_file(runtime / "model.py")
    anchor_records = []''')
    a = xd.index('            if VARIANT == "v7":')
    b = xd.index('            forwards += 1', a)
    xd = xd[:a] + '''            if VARIANT == "v0":
                result = forward(images, padding_mask, text, lengths)
            else:
                result = anchor_forward(
                    self, forward, images, padding_mask, text, lengths,
                    ANCHOR_CONFIG, anchor_records,
                )
                anchor_records[-1]["video_id"] = inventory[forwards]["official_path"]
''' + xd[b:]
    xd = replace_once(xd, '        "metrics": None,',
                     '        "metrics": None, "anchor_config": ANCHOR_CONFIG, **model_receipt,')
    xd = replace_once(xd, '    (output_dir / "run.json").write_text(', '''    (output_dir / "anchor_diagnostics.json").write_text(
        json.dumps(anchor_records, indent=2), encoding="utf-8")
    (output_dir / "run.json").write_text(''')
    xd = replace_once(xd, '        from utils.tools import get_batch_mask\n', '')
    # These evaluators install optional dependencies and add upstream to sys.path
    # before importing it. A main function preserves that order without E402.
    def format_runner(source):
        wrapped = 'def main():\n' + textwrap.indent(source.strip(), '    ') + '\n\nmain()\n'
        proc = subprocess.run(
            [sys.executable, '-m', 'ruff', 'check', '--select', 'I', '--fix',
             '--stdin-filename', 'runner.py', '-'],
            input=wrapped, text=True, encoding='utf-8', capture_output=True, check=True,
        )
        return proc.stdout

    ucf, xd = format_runner(ucf), format_runner(xd)
    config = "".join(base["cells"][1]["source"]) + '''
# Fixed exploratory configuration: do not tune on official test results.
ANCHOR_CONFIG = {
    "beta_max": 0.10, "normal_c_max": 0.20, "normal_a_max": 0.20,
    "bottom_fraction": 0.20, "min_snippets": 8,
    "min_coherence": 0.50, "neighbor_radius": 1,
}
BOOTSTRAP_REPLICATES = 500
SEED = 42
'''
    cells = [cell("markdown", (HERE / "intro.md").read_text(encoding="utf-8")),
             cell("code", config)]
    for name, filename in (("ANCHOR_SOURCE", "anchor.py"), ("ANALYSIS_SOURCE", "analysis.py")):
        cells.append(cell("code", f"{name} = { (HERE / filename).read_text(encoding='utf-8')!r}\n"))
    cells.extend([cell("code", f"UCF_RUNNER = {ucf!r}\n"), cell("code", f"XD_RUNNER = {xd!r}\n"),
                  cell("code", (HERE / "driver.py").read_text(encoding="utf-8"))])
    base["cells"] = cells
    target = NOTEBOOKS / "V16.ipynb"
    target.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    # Keep generated evaluator copies for reproducible local checking.
    for name, source in (("ucf_runner.py", ucf), ("xd_runner.py", xd)):
        (HERE / name).write_text(source, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
