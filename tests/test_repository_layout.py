"""Guard clone-time source contracts after retiring the legacy research modules."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_notebooks_are_standalone_and_compile():
    for version in range(22):
        path = ROOT / f"versions/V{version:02d}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec")


def test_version_index_hashes_pin_actual_notebooks():
    entries = json.loads((ROOT / "tests/fixtures/notebook_hashes.json").read_text(encoding="utf-8"))
    assert {entry["version"] for entry in entries} == {f"V{i:02d}" for i in range(22)}
    for entry in entries:
        path = ROOT / entry["notebook"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_no_legacy_imports_in_active_python():
    for base in (ROOT / "versions/_dev",):
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "from temporal_hoi" not in text
            assert "import temporal_hoi" not in text
