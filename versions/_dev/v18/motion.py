"""Read verified snippet-level motion or extract it from explicitly mapped video."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

FLOW_DIM = 32
SCHEMA = "vadclip_v18_motion_v1"


def feature_id(name):
    stem, sep, crop = PurePosixPath(str(name).replace("\\", "/")).name.rpartition("__")
    if not sep or not stem or crop not in {f"{i}.npy" for i in range(10)}:
        raise ValueError(f"Invalid feature basename: {name}")
    return stem.casefold()


def safe_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes data root: {name}")
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def difference_flow(previous, current, cv2):
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Approximate translation compensation. This does not remove zoom or camera cuts.
    flow -= np.median(flow, axis=(0, 1), keepdims=True)
    out = []
    for row in np.array_split(flow, 2, axis=0):
        for tile in np.array_split(row, 2, axis=1):
            dx, dy = tile.reshape(-1, 2).T
            magnitude = np.hypot(dx, dy)
            bins = np.floor((np.arctan2(dy, dx) + np.pi) * (4 / np.pi)).astype(int) % 8
            hist = np.bincount(bins, weights=np.minimum(magnitude, 10), minlength=8)
            out.extend(np.log1p(hist / len(dx)))
    return np.array(out, dtype=np.float32)


def extract_video(path, spans):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {path}")
    rows = []
    frame_index = -1
    try:
        for start, stop in spans:
            previous = None
            samples = []
            while frame_index < stop - 1:
                ok, frame = cap.read()
                frame_index += 1
                if not ok:
                    raise ValueError(f"Video ended at frame {frame_index}: {path}")
                if frame_index < start or (frame_index - start) % 2:
                    continue
                gray = cv2.cvtColor(cv2.resize(frame, (160, 96), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2GRAY)
                if previous is not None:
                    samples.append(difference_flow(previous, gray, cv2))
                previous = gray
            rows.append(np.mean(samples, axis=0))
    finally:
        cap.release()
    return np.asarray(rows, dtype=np.float32)


class MotionPack:
    def __init__(self, source, dataset, cache):
        source = Path(source)
        self.root = source.parent
        self.archive = ZipFile(source) if source.suffix.lower() == ".zip" else None
        try:
            payload = self.archive.read("manifest.json") if self.archive else source.read_bytes()
        finally:
            if self.archive:
                self.archive.close()
        self.zip_path = source if self.archive else None
        self.sha256 = digest(payload)
        self.cache = Path(cache) / self.sha256
        manifest = json.loads(payload)
        if manifest["schema"] != SCHEMA or manifest["dataset"] != dataset:
            raise ValueError("Motion manifest schema/dataset mismatch.")
        self.items = {}
        for item in manifest["videos"]:
            key = item["id"].casefold()
            if key in self.items or "flow" in item and "video" in item:
                raise ValueError(f"Duplicate or ambiguous motion record: {key}")
            self.items[key] = item

    def verify(self, key, steps, feature_sha):
        key = key.casefold()
        item = self.items.get(key)
        if item is None:
            raise ValueError(f"Motion manifest is missing video {key}.")
        if item["clip_sha256"] != feature_sha or len(item["frame_spans"]) != steps:
            raise ValueError(f"Motion/CLIP hash or snippet count mismatch: {key}")
        spans = np.asarray(item["frame_spans"])
        if (spans.shape != (steps, 2) or spans.dtype.kind not in "iu" or
                spans[0, 0] < 0 or np.any(spans[:, 1] - spans[:, 0] < 3) or
                np.any(spans[1:, 0] < spans[:-1, 1])):
            raise ValueError(f"Invalid zero-based, nonoverlapping frame intervals: {key}")
        if "flow" in item:
            name = PurePosixPath(item["flow"])
            if name.is_absolute() or ".." in name.parts or name.suffix != ".npy":
                raise ValueError(f"Invalid flow path: {key}")
        elif self.zip_path or not safe_path(self.root, item["video"]).is_file():
            raise ValueError(f"Raw video must be mounted next to a JSON manifest: {key}")
        return item, spans

    def array(self, key, steps, feature_sha):
        item, spans = self.verify(key, steps, feature_sha)
        if "flow" in item:
            if self.zip_path:
                with ZipFile(self.zip_path) as archive:
                    data = archive.read(item["flow"])
            else:
                data = safe_path(self.root, item["flow"]).read_bytes()
            if digest(data) != item["flow_sha256"]:
                raise ValueError(f"Flow SHA-256 mismatch: {key}")
            flow = np.load(io.BytesIO(data), allow_pickle=False)
        else:
            cache = self.cache / f"{digest(key.encode())}.npy"
            if not cache.exists():
                video = safe_path(self.root, item["video"])
                with video.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != item["video_sha256"]:
                        raise ValueError(f"Raw video SHA-256 mismatch: {key}")
                flow = extract_video(video, spans)
                self.cache.mkdir(parents=True, exist_ok=True)
                pending = cache.with_suffix(".pending")
                with pending.open("wb") as handle:
                    np.save(handle, flow, allow_pickle=False)
                pending.replace(cache)
            flow = np.load(cache, allow_pickle=False)
        if (flow.shape != (steps, FLOW_DIM) or flow.dtype.kind != "f" or
                not np.isfinite(flow).all() or (flow < 0).any()):
            raise ValueError(f"Invalid motion features (expected nonnegative [{steps}, 32]): {key}")
        return flow.astype(np.float32, copy=False)
