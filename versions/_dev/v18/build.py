"""Build self-contained V18 from the verified V7 evaluator and V14 split code."""

import ast
import json
from pathlib import Path

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"



HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return path.read_text(encoding="utf-8")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Missing/ambiguous source marker: {old}")
    return text.replace(old, new)


def embedded(cell):
    return ast.literal_eval(ast.parse("".join(cell["source"])).body[0].value)


def build():
    notebook = json.loads((NOTEBOOKS / "V07.ipynb").read_bytes())
    cells = notebook["cells"]
    ucf, xd = (embedded(cells[i]) for i in (5, 6))
    ucf = replace_once(ucf, 'if VARIANT == "v7":', 'if VARIANT != "v0":')
    ucf = replace_once(ucf, '        c_score = ', '''        logits_c, logits_a = apply_head(
            bundle, visual, lengths, logits_c, logits_a, "C")
        c_score = ''')
    ucf = replace_once(ucf, '\n# Chọn theo thứ tự', '''
from head import apply_head, load_head

bundle = load_head(HEAD_PATH, STRENGTH, USE_TEMPORAL, device) if HEAD_PATH else None

# Chọn theo thứ tự''')
    xd = replace_once(xd, 'if VARIANT == "v7":', 'if VARIANT != "v0":')
    xd = replace_once(xd, "        original_forward = CLIPVAD.forward", '''        from head import apply_head, load_head

        bundle = load_head(HEAD_PATH, STRENGTH, USE_TEMPORAL, torch.device("cuda:0")) if HEAD_PATH else None
        original_forward = CLIPVAD.forward''')
    xd = replace_once(xd, "            forwards += 1", '''            if bundle is not None:
                t, c, a = result
                c, a = apply_head(bundle, images, lengths, c, a, "A")
                result = t, c, a
            forwards += 1''')

    legacy = ROOT / "templates"
    trainer = read(legacy / "trainer.py.txt")
    start, end = trainer.index("from v14_runtime_support import"), trainer.index("COMMIT =")
    fit = read(HERE / "fit.py")
    imports, functions = fit[fit.index("from head import"):].split("def cache_v7(", 1)
    trainer = trainer[:start] + imports + trainer[end:]
    start, end = trainer.index("def trainable_fingerprint("), trainer.index("def save_checkpoint(")
    trainer = trainer[:start] + '''def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


''' + "def cache_v7(" + functions + "\n\n" + trainer[end:]
    trainer = trainer.replace("V14", "V18").replace("vadclip_v14_source_", "vadclip_v18_source_")
    v17 = read(ROOT / "v17" / "driver.py")
    tables = v17[v17.index("def format_table("):v17.index("def run_both(")]
    tables = replace_once(tables, '"v0": "V0", "v7": "V7", "residual_only": "Residual riêng", "v17": "V17"',
                          '"v7": "V7", "clip_only": "CLIP", "temporal_random": "CLIP + temporal", "v18": "V18"')
    tables = tables.replace('row["comparison"] == "v17_vs_v7"', 'row["comparison"] == "v18_vs_v7"')
    analysis = read(ROOT / "v16" / "analysis.py")
    coverage = read(legacy / "v14_runtime_support.py")
    analysis += "\n\nimport math\n\n" + coverage[
        coverage.index("def event_intervals("):coverage.index("def duration_diagnostics(")]
    helper = read(legacy / "driver.py").split("def validate_v14_config(", 1)[0]
    config = "".join(cells[1]["source"]) + "\n" + read(HERE / "config.py")
    sources = {
        "v7_windows.py": embedded(cells[3]), "head.py": read(HERE / "head.py"),
        "analysis.py": analysis,
    }
    contents = [
        ("markdown", read(HERE / "intro.md")),
        ("code", config),
        ("code", f"SOURCES = {sources!r}\nTRAINER_SOURCE = {trainer!r}\n"
         f"UCF_RUNNER = {ucf!r}\nXD_RUNNER = {xd!r}\n"),
        ("code", helper + tables + read(HERE / "driver.py")),
    ]
    for name, source in {**sources, "trainer": trainer, "ucf": ucf, "xd": xd}.items():
        compile(source, name, "exec")
    new = []
    for index, (kind, body) in enumerate(contents):
        cell = dict(cell_type=kind, id=f"v18-cell-{index}", metadata={},
                    source=body.splitlines(keepends=True))
        if kind == "code":
            compile(body, f"cell-{index}", "exec")
            cell.update(execution_count=None, outputs=[])
        new.append(cell)
    notebook["cells"] = new
    target = NOTEBOOKS / "V18.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    build()
