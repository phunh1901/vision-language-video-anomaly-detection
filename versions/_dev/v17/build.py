"""Build the self-contained Kaggle V17 notebook from the V7 evaluator."""

import ast
import json
from pathlib import Path

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"



HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def source(cell):
    return "".join(cell["source"])


def embedded(cell):
    return ast.literal_eval(ast.parse(source(cell)).body[0].value)


def replace_once(original, before, after):
    if original.count(before) != 1:
        raise ValueError(f"Expected one matching V7 fragment: {before!r}")
    return original.replace(before, after, 1)


def cell(kind, contents):
    result = dict(cell_type=kind, metadata={}, source=contents.splitlines(keepends=True))
    if kind == "code":
        compile(contents, "<notebook cell>", "exec")
        result.update(execution_count=None, outputs=[])
    return result


def build():
    notebook = json.loads((NOTEBOOKS / "V07.ipynb").read_bytes())
    cells = notebook["cells"]
    window_source = embedded(cells[3])
    ucf, xd = (embedded(cells[index]) for index in (5, 6))
    for name in ("ucf", "xd"):
        runner = ucf if name == "ucf" else xd
        runner = replace_once(runner, 'if VARIANT == "v7":', 'if VARIANT in {"v7", "v17"}:')
        if name == "ucf":
            runner = replace_once(
                runner, "from v7_windows import overlap_forward",
                "from v7_windows import overlap_forward\nfrom v17_residual import residual_logits",
            )
            runner = replace_once(
                runner, "        c_score = ",
                '        if VARIANT in {"residual_only", "v17"}:\n'
                "            logits_a = residual_logits(logits_c, logits_a, RESIDUAL_ETA)\n"
                "        c_score = ",
            )
            ucf = runner
        else:
            runner = replace_once(
                runner, "    from v7_windows import overlap_forward",
                "    from v7_windows import overlap_forward\n"
                "    from v17_residual import residual_logits",
            )
            runner = replace_once(
                runner, "            forwards += 1",
                '            if VARIANT in {"residual_only", "v17"}:\n'
                '                if not text or str(text[0]).strip().casefold() != "normal":\n'
                '                    raise ValueError("Expected Normal at class index 0.")\n'
                "                text_features, logits_c, logits_a = result\n"
                "                result = (text_features, logits_c,\n"
                "                          residual_logits(logits_c, logits_a, RESIDUAL_ETA))\n"
                "            forwards += 1",
            )
            xd = runner
        compile(runner, f"<{name} evaluator>", "exec")

    library = (ROOT / "v16" / "analysis.py").read_text(encoding="utf-8")
    baseline_driver = (ROOT / "v16" / "driver.py").read_text(encoding="utf-8")
    shared = baseline_driver[:baseline_driver.index("V0_REFERENCE = ")]
    csv_writer = baseline_driver[baseline_driver.index("def write_csv("):
                                 baseline_driver.index("def run_both(")]
    config = source(cells[1]) + '''
# Fixed before reading the V17 official-test result.
RESIDUAL_ETA = 0.10
BOOTSTRAP_REPLICATES = 500
SEED = 42
INCLUDE_PREDICTIONS = False
'''
    notebook["cells"] = [
        cell("markdown", (HERE / "intro.md").read_text(encoding="utf-8")),
        cell("code", config),
        cell("code", "".join(f"{key} = {value!r}\n" for key, value in {
            "WINDOW_SOURCE": window_source,
            "RESIDUAL_SOURCE": (HERE / "residual.py").read_text(encoding="utf-8"),
            "ANALYSIS_SOURCE": library, "UCF_RUNNER": ucf, "XD_RUNNER": xd,
        }.items())),
        cell("code", shared + csv_writer + (HERE / "driver.py").read_text(encoding="utf-8")),
    ]
    path = NOTEBOOKS / "V17.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path.name)


if __name__ == "__main__":
    build()
