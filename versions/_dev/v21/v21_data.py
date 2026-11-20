"""Reused pinned train-pack validation/group split from V20; no test-label tuning."""
import csv
import hashlib
import io
import json
import random
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

SRC_REPO = None  # Set explicitly by the trainer before checkout().

COMMIT = "c41067f07d252efcda18008bea367886070c33b0"
CLIP_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
UCF_CLASSES = (
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
)
XD_LABELS = {
    "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
    "B5": "abuse", "B6": "car accident", "G": "explosion",
}


def file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_basename(path):
    return PurePosixPath(str(path).replace(chr(92), "/")).name


def video_id(path):
    stem, sep, crop = source_basename(path).rpartition("__")
    if not stem or not sep or crop not in {f"{i}.npy" for i in range(10)}:
        raise ValueError(f"Tên feature VadCLIP sai: {path}")
    return stem.casefold()


def checkout():
    if SRC_REPO is None:
        raise ValueError("SRC_REPO must be configured before checkout")
    if not SRC_REPO.exists():
        SRC_REPO.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SRC_REPO.parent) as tmp:
            clone = Path(tmp) / "source"
            subprocess.run(["git", "clone", "https://github.com/nwpu-zxr/VadCLIP.git", str(clone)], check=True)
            subprocess.run(["git", "-C", str(clone), "checkout", "--detach", COMMIT], check=True)
            clone.rename(SRC_REPO)
    actual = subprocess.check_output(["git", "-C", str(SRC_REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(SRC_REPO), "status", "--porcelain", "--untracked-files=all"], text=True,
    ).strip()
    if actual != COMMIT or dirty:
        raise RuntimeError("Cần checkout VadCLIP sạch tại commit đã khóa.")


def official_csv(default, repo):
    path = repo / "list" / source_basename(default)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"path", "label"} <= set(reader.fieldnames):
            raise ValueError(f"Thiếu cột path/label: {path.name}")
        rows = list(reader)
    return rows, file_sha(path)


class FeaturePack:
    def __init__(self, root, prefix, expected_manifest):
        name = f"vadclip_{prefix}_train_v5"
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.rglob(f"{name}_manifest.json"))
            if not candidates:
                candidates = sorted(root.rglob(f"{name}.zip"))
        if len(candidates) != 1:
            raise ValueError(f"Cần đúng một gói {name}.zip hoặc manifest giải nén. Tìm thấy {len(candidates)}.")
        location = candidates[0]
        self.archive = None
        self.directory = location.parent
        manifest_name = f"{name}_manifest.json"
        if location.suffix == ".zip":
            self.archive = ZipFile(location)
            payload = self.archive.read(manifest_name)
        else:
            payload = location.read_bytes()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        if self.manifest_sha256 != expected_manifest:
            raise ValueError(f"Sai hash manifest gói {name}.")
        content = json.loads(payload)
        crop, count = (5, 1610) if prefix == "ucf" else (0, 3954)
        if (content["dataset"] != prefix or content["crop"] != crop
                or content["schema"] != "vadclip_v5_train_v1" or len(content["features"]) != count):
            raise ValueError(f"Sai schema hoặc số video trong gói {name}.")
        self.features = {}
        for idx, item in enumerate(content["features"], 1):
            if item["path"] != f"features/{prefix}_{idx:05d}.npy":
                raise ValueError("Tên vật lý trong manifest train không hợp lệ.")
            key = video_id(item["original"])
            if key in self.features:
                raise ValueError(f"ID trùng trong gói train: {key}")
            self.features[key] = item
        self.checked = set()

    def array(self, item):
        path = item["path"]
        data = self.archive.read(path) if self.archive else (self.directory / path).read_bytes()
        if path not in self.checked:
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"SHA-256 sai cho feature train: {path}")
            self.checked.add(path)
        array = np.load(io.BytesIO(data), allow_pickle=False)
        if (array.shape != (item["steps"], 512) or not len(array)
                or array.dtype.kind != "f" or not np.isfinite(array).all()):
            raise ValueError(f"Shape/dtype/giá trị feature train sai: {path}")
        return array

    def close(self):
        if self.archive is not None:
            self.archive.close()


def select_train_rows(train_rows, features, encode_labels):
    """Gộp các crop của cùng một video, ưu tiên crop có trong gói feature."""
    selected = {}
    for row in train_rows:
        key = video_id(row["path"])
        feature = features.get(key)
        matches_pack_crop = (
            feature is not None
            and source_basename(row["path"]).casefold()
            == source_basename(feature["original"]).casefold()
        )
        previous = selected.get(key)
        if previous is not None:
            previous_row, previous_matches = previous
            if previous_row["label"] != row["label"]:
                encoded = encode_labels([previous_row["label"], row["label"]])
                if not torch.equal(encoded[0], encoded[1]):
                    raise ValueError(
                        f"Nhãn khác nhau giữa các crop của video {key}: "
                        f"{previous_row['label']} và {row['label']}."
                    )
            if previous_matches or not matches_pack_crop:
                continue
        selected[key] = (row, matches_pack_crop)
    return [row for row, _ in selected.values()]


def dataset_split(train_rows, test_rows, features, encode_labels, prefix, seed,
                  protocol="source_disjoint_development", preserve_crops=False):
    train_videos = [video_id(row["path"]) for row in train_rows]
    train_ids = ([Path(source_basename(row["path"])).stem.casefold() for row in train_rows]
                 if preserve_crops else train_videos)
    test_ids = {video_id(row["path"]) for row in test_rows}
    if set(train_videos) & test_ids:
        raise ValueError("ID official train giao official test.")
    if set(train_ids) != features.keys():
        missing = set(train_ids) - features.keys()
        extra = features.keys() - set(train_ids)
        raise ValueError(f"Gói không khớp CSV train. Thiếu {len(missing)}, thừa {len(extra)}. Mẫu: {sorted(missing)[:4]}.")
    labels = encode_labels([row["label"] for row in train_rows]).float().cpu()
    if labels.ndim != 2 or (labels.sum(1) <= 0).any():
        raise ValueError("Nhãn video train có hàng rỗng.")
    abnormal = (labels[:, 0] == 0).numpy()
    if (labels[:, 0].bool() & labels[:, 1:].sum(1).bool()).any():
        raise ValueError("Nhãn Normal đồng thời có nhãn bất thường.")
    if (labels.sum(0) <= 0).any():
        raise ValueError("Official train thiếu lớp trong label map.")
    groups = [vid.split("__#", 1)[0] if prefix == "xd" else vid for vid in train_videos]
    test_groups = {vid.split("__#", 1)[0] if prefix == "xd" else vid for vid in test_ids}
    if protocol == "official_train_final_epoch":
        # The official ID split is retained. Source overlap is disclosed, not silently removed.
        return (train_ids, groups, labels, abnormal, np.arange(len(train_ids)),
                np.array([], dtype=int), np.array([], dtype=int), len(set(groups) & test_groups))
    if protocol != "source_disjoint_development":
        raise ValueError(f"Unknown training protocol: {protocol}")
    excluded = np.flatnonzero(np.isin(groups, sorted(test_groups)))
    eligible = np.flatnonzero(~np.isin(groups, sorted(test_groups)))
    if len(eligible) < 2 or (labels[eligible].sum(0) <= 0).any():
        raise ValueError("Lọc nhóm tên nguồn khiến train/dev thiếu dữ liệu hoặc thiếu lớp.")
    eligible_groups = np.asarray(groups)[eligible]
    candidates = GroupShuffleSplit(n_splits=30, test_size=0.10, random_state=seed)
    for train_positions, dev_positions in candidates.split(eligible, abnormal[eligible], eligible_groups):
        train, dev = eligible[train_positions], eligible[dev_positions]
        if (len(np.unique(abnormal[train])) == 2 and len(np.unique(abnormal[dev])) == 2
                and bool((labels[train].sum(0) > 0).all())):
            assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[dev])
            return train_ids, groups, labels, abnormal, train, dev, excluded, len(set(groups) & test_groups)
    raise ValueError("Chưa tìm được train/development đủ hai nhóm và độ phủ train.")


def balanced_batches(indices, abnormal, half, seed):
    rng = np.random.default_rng(seed)
    positive = rng.permutation(indices[abnormal[indices]])
    normal = rng.permutation(indices[~abnormal[indices]])
    batches = (max(len(positive), len(normal)) + half - 1) // half
    positive = np.resize(positive, batches * half).reshape(batches, half)
    normal = np.resize(normal, batches * half).reshape(batches, half)
    return np.concatenate((normal, positive), axis=1)





def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model, path):
    pending = path.with_suffix(".pth.partial")
    torch.save(model.state_dict(), pending)
    pending.replace(path)


