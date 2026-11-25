"""Verify the public boundary without opening any account or secret file."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ignored(paths):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        input=("\0".join(paths) + "\0").encode(), capture_output=True, cwd=ROOT, check=False,
    )
    assert result.returncode in {0, 1}, result.stderr
    return {path.decode() for path in result.stdout.split(b"\0") if path}


def test_private_documents_and_binary_output_are_excluded_at_any_depth():
    paths = [
        "AGENTS.md", "plan.md", "rules.md", "rubric.md", "HUONG_DAN_TIEN_DO.md",
        ".codex/config.toml", ".aws/credentials", ".agents/rules/readme_conventions.md",
        "reports/private.md", "docs/prd.md", "data/input.zip", "weights/model.pth",
        "versions/_dev/v21/AGENTS.md", "versions/_dev/v21/plan.md",
        "versions/_dev/v21/.env", "versions/_dev/v21/kaggle.json",
        "results/V21/run_c5ieenya/result.zip", "results/V21/run_c5ieenya/run.json",
        "results/V21/run_c5ieenya/metrics.npz", "results/V21/run_c5ieenya/console.log",
        "versions/_dev/data/train.json", "versions/_dev/datasets/video.mp4",
        "versions/_dev/features/cached.npy", "versions/_dev/weights/model.pth",
        "versions/_dev/train.parquet", "versions/_dev/annotations.jsonl",
        "versions/_dev/frames/frame.jpg", "tests/data/labels.tsv",
    ]
    assert ignored(paths) == set(paths)


def test_code_and_reviewed_result_text_are_eligible():
    paths = [
        "README.md", ".gitignore", ".gitattributes", "pyproject.toml", "uv.lock",
        "versions/V00.ipynb", "versions/V07.ipynb", "versions/V20.ipynb",
        "tests/test_repository_layout.py", "tests/fixtures/notebook_hashes.json",
        "results/README.md", "results/V07/run_tutkhmvv/notes.md",
        "results/V20/run_259y0a3x/metrics.json",
    ]
    assert ignored(paths) == set()
