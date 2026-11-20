"""Expose local version sources for CPU regression tests without data downloads."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in [ROOT / "versions/_dev/common/clip_vad", ROOT / "versions/_dev/v16",
                  ROOT / "versions/_dev/v21"]:
    sys.path.insert(0, str(directory))
