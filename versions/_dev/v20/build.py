"""Build the V20 notebook from the tested V19 Kaggle runtime."""

import json
from pathlib import Path

REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = REPOSITORY / "versions"
SHARED = REPOSITORY / "versions/_dev/common/clip_vad"



HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"V19 source marker missing or ambiguous: {old!r}")
    return source.replace(old, new)


def build():
    notebook = json.loads((NOTEBOOKS / "V19.ipynb").read_bytes())
    cells = notebook["cells"]
    payload = {}
    exec("".join(cells[2]["source"]), payload)
    sources = payload["SOURCES"]
    head = sources["head.py"]
    center = (HERE / "center.py.txt").read_text(encoding="utf-8")
    head = replace_once(head, "def load_head(", center + "def load_head(")
    head = replace_once(head, "use_temporal, device):", "use_temporal, device, centered=False):")
    head = replace_once(head, "return head.eval(), strength, use_temporal",
                        "return head.eval(), strength, use_temporal, centered")
    head = replace_once(head, "head, strength, use_temporal = bundle",
                        "head, strength, use_temporal, centered = bundle")
    head = replace_once(head, "    delta = gated_delta(raw, support, strength)",
                        "    correction = centered_delta if centered else gated_delta\n"
                        "    delta = correction(raw, support, strength)")
    sources["head.py"] = head.replace("Non-finite V19 residual", "Non-finite V20 residual")

    trainer = payload["TRAINER_SOURCE"]
    start = trainer.index("    if result[\"normal_only\"]")
    end = trainer.index("    return result, receipt", start)
    trainer = trainer[:start] + trainer[end:]
    trainer = trainer.replace('def fit_one(context, schedule, dev, cache, variant):',
                              'def fit_one(context, schedule, dev, cache):')
    trainer = trainer.replace('    total_pairs = replaced = 0', '    total_pairs = 0')
    trainer = trainer.replace(
        '            scores, positives, normals, backgrounds, weights = [], [], [], [], []',
        '            scores, positives, normals, weights = [], [], [], []',
    )
    trainer = trainer.replace('                    pos, neg, weight = extra[0]', '                    pos, _, weight = extra[0]')
    trainer = trainer.replace('                        ix = torch.as_tensor(neg.clip(min=0), device=device)\n', '')
    trainer = trainer.replace('                        backgrounds.append((raw[ix], torch.as_tensor(neg >= 0, device=device)))\n', '')
    trainer = trainer.replace('                bg, valid = (torch.cat(parts) for parts in zip(*backgrounds, strict=True))\n', '')
    trainer = trainer.replace('                if variant == "v19":\n'
                              '                    sampled = torch.where(valid, bg, sampled)\n'
                              '                    replaced += int(valid.sum())\n', '')
    trainer = trainer.replace('print(f"{PREFIX}/{variant}: epoch', 'print(f"{PREFIX}/normal_only: epoch')
    trainer = trainer.replace('f"pairs={total_pairs}, replaced={replaced}"', 'f"pairs={total_pairs}"')
    trainer = trainer.replace('path = OUTPUT / f"{variant}.pth"', 'path = OUTPUT / "normal_only.pth"')
    trainer = trainer.replace('"replaced_pairs": replaced,', '"replaced_pairs": 0,')
    trainer = trainer.replace('result = {variant: fit_one(context, schedule, dev, cache, variant) for variant in VARIANTS}',
                              'result = {"normal_only": fit_one(context, schedule, dev, cache)}')
    trainer = trainer.replace("V19", "V20").replace("vadclip_v19_source_", "vadclip_v20_source_")
    assert "replaced +=" not in trainer and "for variant in VARIANTS" not in trainer

    ucf = replace_once(payload["UCF_RUNNER"], 'USE_TEMPORAL, device) if HEAD_PATH',
                       'USE_TEMPORAL, device, centered=VARIANT == "v20") if HEAD_PATH')
    xd = replace_once(payload["XD_RUNNER"], 'torch.device("cuda:0")) if HEAD_PATH',
                      'torch.device("cuda:0"), centered=VARIANT == "v20") if HEAD_PATH')

    driver = "".join(cells[3]["source"]).replace("v19", "v20").replace("V19", "V20")
    driver = replace_once(driver, '"v7": "V7", "normal_only": "Temporal (Normal)"',
                          '"v0": "V0", "v7": "V7", "normal_only": "V7 + temporal"')
    driver = replace_once(driver, 'row["comparison"] == "v20_vs_v7"', 'row["comparison"] == "v20_vs_v0"')
    driver = replace_once(driver, 'VARIANTS = ("v7", "normal_only", "v20")',
                          'VARIANTS = ("v0", "v7", "normal_only", "v20")')
    driver = replace_once(driver, 'PAIRS = (("v7", "normal_only"), ("normal_only", "v20"), ("v7", "v20"))',
                          'PAIRS = (("v0", "v7"), ("v7", "normal_only"), ("normal_only", "v20"), '
                          '("v7", "v20"), ("v0", "v20"))')
    driver = replace_once(driver, '    for name, current in arrays.items():',
                          '''    original = reports["v0"]["metrics"][primary_branch][primary_metric]
    if abs(original - V0_REFERENCE[prefix][primary_metric]) > 0.00005:
        raise ValueError("V0 reproduction check failed.")
    feature_hash = "feature_inventory_sha256" if prefix == "ucf" else "source_features_archive_sha256"
    for name, current in arrays.items():''')
    driver = replace_once(driver, '"frames_per_feature", "visual_block_length")',
                          '"frames_per_feature", "visual_block_length", feature_hash)')
    driver = replace_once(driver, '        if not np.array_equal(current["scores"][:, 1-primary_column],',
                          '        if name != "v0" and not np.array_equal(current["scores"][:, 1-primary_column],')
    driver = replace_once(driver, '        for name, data in arrays.items()',
                          '        for name, data in arrays.items() if name != "v0"')
    driver = replace_once(driver, 'entry.update(coverage_v20_vs_v7=coverage, changed_snippets=changed)',
                          'entry.update(coverage_v20_vs_v7=coverage, changed_snippets_vs_v7=changed)')
    start = driver.index('            dest = output / prefix / "v7"')
    end = driver.index('            run_dataset(f"{prefix}_train"', start)
    driver = driver[:start] + '''            for variant in ("v0", "v7"):
                dest = output / prefix / variant
                run_dataset(f"{prefix}_{variant}", runner, {
                    **evaluation, "OUTPUT": dest, "VARIANT": variant,
                    "HEAD_PATH": None, "STRENGTH": 0., "USE_TEMPORAL": False,
                }, scripts)
                paths[variant] = report_path(dest)
                entry["reports"][variant] = paths[variant].relative_to(output).as_posix()
''' + driver[end:]
    driver = replace_once(driver, '"VARIANTS": VARIANTS[1:], "TEMPORAL_CONFIG": TEMPORAL_CONFIG,',
                          '"TEMPORAL_CONFIG": TEMPORAL_CONFIG,')
    driver = replace_once(driver, '''            for variant in VARIANTS[1:]:
                record = trained["variants"][variant]
                weights = train_output / f"{variant}.pth"
                if file_sha(weights) != record["checkpoint_sha256"]:
                    raise ValueError(f"Head checkpoint hash mismatch: {prefix}/{variant}")
''', '''            record = trained["variants"]["normal_only"]
            weights = train_output / record["checkpoint_file"]
            if file_sha(weights) != record["checkpoint_sha256"]:
                raise ValueError(f"Head checkpoint hash mismatch: {prefix}/normal_only")
            entry["shared_temporal_checkpoint_sha256"] = record["checkpoint_sha256"]
            for variant in ("normal_only", "v20"):
''')
    config = "".join(cells[1]["source"])
    contents = [(HERE / "intro.md").read_text(encoding="utf-8"), config,
                f"SOURCES = {sources!r}\nTRAINER_SOURCE = {trainer!r}\nUCF_RUNNER = {ucf!r}\nXD_RUNNER = {xd!r}\n",
                driver]
    for label, code in {**sources, "trainer": trainer, "ucf": ucf, "xd": xd,
                        "notebook": "\n".join(contents[1:])}.items():
        compile(code, label, "exec")
    for i, (cell, code) in enumerate(zip(cells, contents, strict=True)):
        cell.update(id=f"v20-cell-{i}", source=code.splitlines(keepends=True))
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    target = NOTEBOOKS / "V20.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    build()
