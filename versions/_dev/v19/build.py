"""Build the V19 Kaggle notebook using the validated V18 runtime."""

import json
from pathlib import Path

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"



HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def source(cell):
    return "".join(cell["source"])


def build():
    notebook = json.loads((NOTEBOOKS / "V18.ipynb").read_bytes())
    cells = notebook["cells"]
    payload = {}
    exec(source(cells[2]), payload)
    sources = payload["SOURCES"]
    sources["head.py"] = sources["head.py"].split("def load_head(", 1)[0] + (
        HERE / "head_tail.py.txt"
    ).read_text(encoding="utf-8")
    trainer = payload["TRAINER_SOURCE"]
    trainer = (trainer[:trainer.index("from head import")]
               + (HERE / "fit.py.txt").read_text(encoding="utf-8")
               + trainer[trainer.index("COMMIT ="):trainer.index("def cache_v7(")]
               + trainer[trainer.index("def save_checkpoint("):])
    trainer = trainer.replace("V18", "V19").replace("vadclip_v18_source_", "vadclip_v19_source_")

    config = source(cells[1]).replace('{"epochs": 3}', '{"epochs": 3, "max_logit_delta": 0.25}')
    driver = source(cells[3]).replace("v18", "v19").replace("V18", "V19")
    driver = driver.replace(
        'VARIANTS = ("v7", "clip_only", "temporal_random", "v19")',
        'VARIANTS = ("v7", "normal_only", "v19")',
    )
    driver = driver.replace(
        'PAIRS = (("v7", "clip_only"), ("clip_only", "temporal_random"),\n'
        '         ("temporal_random", "v19"), ("v7", "v19"))',
        'PAIRS = (("v7", "normal_only"), ("normal_only", "v19"), ("v7", "v19"))',
    )
    driver = driver.replace(
        '"clip_only": "CLIP", "temporal_random": "CLIP + temporal"',
        '"normal_only": "Temporal (Normal)"',
    )
    driver = driver.replace(
        '"STRENGTH": 1. if RUN_MODE == "smoke" else record["strength"],\n'
        '                    "USE_TEMPORAL": variant != "clip_only",',
        '"STRENGTH": record["strength"], "USE_TEMPORAL": True,',
    )
    driver = driver.replace('for variant in ("v7", "v19"):', 'for variant in VARIANTS:')
    driver = driver.replace(
        '    coverage = analysis.compare_coverage(',
        '''    changed = {
        name: {
            "changed_snippets": int(np.count_nonzero(data["scores"][:, primary_column] !=
                                                     base["scores"][:, primary_column])),
            "total_snippets": int(offsets[-1]),
        }
        for name, data in arrays.items()
    }
    coverage = analysis.compare_coverage(''',
    ).replace("return rows, coverage", "return rows, coverage, changed")
    driver = driver.replace(
        '                metrics, entry["coverage_v19_vs_v7"] = compare_variants(prefix, paths, analysis)',
        '                metrics, coverage, changed = compare_variants(prefix, paths, analysis)\n'
        '                entry.update(coverage_v19_vs_v7=coverage, changed_snippets=changed)',
    )
    driver = driver.replace('    if BOOTSTRAP_REPLICATES < 1 or not INPUT.is_dir():',
                            '    if not 0 < TEMPORAL_CONFIG["max_logit_delta"] <= 0.25:\n'
                            '        raise ValueError("max_logit_delta must be in (0, 0.25].")\n'
                            '    if BOOTSTRAP_REPLICATES < 1 or not INPUT.is_dir():')
    assert "clip_only" not in driver and "temporal_random" not in driver
    assert "record[\"strength\"]" in driver
    ucf, xd = payload["UCF_RUNNER"], payload["XD_RUNNER"]
    contents = [
        (HERE / "intro.md").read_text(encoding="utf-8"),
        config,
        f"SOURCES = {sources!r}\nTRAINER_SOURCE = {trainer!r}\nUCF_RUNNER = {ucf!r}\nXD_RUNNER = {xd!r}\n",
        driver,
    ]
    for label, text in {**sources, "trainer": trainer, "ucf": ucf, "xd": xd,
                        "notebook": "\n".join(contents[1:])}.items():
        compile(text, label, "exec")
    for i, (cell, text) in enumerate(zip(cells, contents, strict=True)):
        cell.update(id=f"v19-cell-{i}", source=text.splitlines(keepends=True))
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    target = NOTEBOOKS / "V19.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    build()
