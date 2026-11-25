"""Expose shared evaluation code to the CPU regression tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "versions/_dev/common/clip_vad"))
