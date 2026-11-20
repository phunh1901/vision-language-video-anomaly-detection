"""CPU crop validation and explicit pinned-repository cosine recipe.

Full-crop inputs must contain the original named __0.npy ... __9.npy files.
Missing views are never manufactured by repeating a single cached crop.
"""

import ast
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
import v21_data as data

PINNED_DEFAULTS = {
    "ucf": {"epochs": 10, "learning_rate": 2e-5, "batch_size": 128,
            "scheduler_milestones": [4, 8], "scheduler_gamma": .1},
    "xd": {"epochs": 10, "learning_rate": 1e-5, "batch_size": 96,
           "scheduler_milestones": [3, 6, 10], "scheduler_gamma": .1},
}


def crop_id(path):
    return Path(data.source_basename(path)).stem.casefold()


def discover_train_root(override, input_root, prefix, rows, output):
    """Resolve complete original crops; always save a useful input inventory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if override is not None:
        return Path(override)
    input_root = Path(input_root)
    expected = {data.source_basename(row["path"]).casefold() for row in rows}
    candidates, inventory = [], []
    raw = {p.name.casefold() for p in input_root.rglob("*.npy")}
    matched = len(expected & raw)
    if matched:
        inventory.append({"path": str(input_root), "kind": "extracted_original_files",
                          "matching_original_crops": matched, "required_crops": len(expected)})
        if expected <= raw:
            candidates.append(input_root)
    legacy_name = f"vadclip_{prefix}_train_v5_manifest.json"

    def record_legacy(manifest, location):
        if manifest.get("schema") == "vadclip_v5_train_v1" and manifest.get("dataset") == prefix:
            inventory.append({"path": str(location), "kind": "legacy_one_crop_pack",
                              "available_crop_features": len(manifest["features"]),
                              "unique_videos": len(manifest["features"]), "crops_per_video": 1,
                              "crop_index": manifest["crop"], "required_crops": len(expected)})

    for path in sorted(input_root.rglob("*.zip")):
        try:
            with ZipFile(path) as archive:
                names = archive.namelist()
                keys = {data.source_basename(name).casefold() for name in names}
                matching = len(expected & keys)
                if matching:
                    inventory.append({"path": str(path), "kind": "original_feature_zip",
                                      "matching_original_crops": matching, "required_crops": len(expected)})
                    if expected <= keys:
                        candidates.append(path)
                if legacy_name in names:
                    record_legacy(json.loads(archive.read(legacy_name)), path)
        except (BadZipFile, OSError, ValueError) as exc:
            inventory.append({"path": str(path), "kind": "unreadable_archive", "error": str(exc)})
    for path in sorted(input_root.rglob(legacy_name)):
        record_legacy(json.loads(path.read_text(encoding="utf-8")), path)
    report = {"dataset": prefix, "automatic_discovery": True, "search_root": str(input_root),
              "required_crop_features": len(expected), "candidates": list(map(str, candidates)),
              "inputs_found": inventory, "gpu_work_started": False}
    (output / "input_discovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Train input discovery:", json.dumps(report, indent=2), flush=True)
    if not candidates:
        legacy = [item for item in inventory if item["kind"] == "legacy_one_crop_pack"]
        detail = (f" Found one-crop v5 pack(s): {[(x['unique_videos'], x['crop_index']) for x in legacy]}."
                  if legacy else "")
        raise ValueError(
            f"{prefix}: no complete 10-crop training input found under {input_root}; "
            f"need {len(expected)} original crop files.{detail} "
            "See input_discovery.json for attached inputs. Attach the original ten-crop features "
            f"or set {prefix.upper()}_TRAIN_ROOT if they are outside this directory. No GPU work started.")
    if len(candidates) != 1:
        raise ValueError(f"Multiple complete {prefix} inputs found: {candidates}; set {prefix.upper()}_TRAIN_ROOT")
    return candidates[0]


class FullCropPack:
    """Read original feature names directly from one ZIP or extracted directory."""

    def __init__(self, root, prefix, rows):
        if root is None:
            raise ValueError(
                f"Set {prefix.upper()}_TRAIN_ROOT to the original 10-crop train ZIP or "
                "extracted folder. Existing vadclip_*_train_v5 packs contain only one "
                "crop/video and cannot reproduce the native exposure. No GPU work started.")
        root = Path(root)
        self.archive = None
        self.directory = root
        self.checked = set()
        self.features = {}
        self.manifest_sha256 = None
        if root.is_file():
            if root.suffix.lower() != ".zip":
                raise ValueError("Full crop input must be a ZIP or an extracted directory")
            self.archive = ZipFile(root)
            names = [x.filename for x in self.archive.infolist() if not x.is_dir()]
        elif root.is_dir():
            names = [x.relative_to(root).as_posix() for x in root.rglob("*.npy")]
        else:
            raise FileNotFoundError(root)
        try:
            legacy_name = f"vadclip_{prefix}_train_v5_manifest.json"
            legacy = None
            if self.archive and legacy_name in names:
                legacy = json.loads(self.archive.read(legacy_name))
            elif root.is_dir() and (root / legacy_name).is_file():
                legacy = json.loads((root / legacy_name).read_text(encoding="utf-8"))
            if legacy and legacy.get("schema") == "vadclip_v5_train_v1":
                raise ValueError(
                    f"{prefix}: supplied v5 pack has {len(legacy['features'])} videos, "
                    f"ONE crop/video (index {legacy['crop']}). Need all TEN original "
                    "crop files/video; no GPU work started.")
            expected = {data.source_basename(row["path"]).casefold(): row for row in rows}
            if len(expected) != len(rows):
                raise ValueError("Official train CSV has duplicate crop filenames")
            found = {}
            for name in names:
                key = data.source_basename(name).casefold()
                if key not in expected:
                    continue
                if key in found:
                    raise ValueError(f"Ambiguous duplicate train feature: {key}")
                found[key] = name
            missing = sorted(set(expected) - set(found))
            if missing:
                raise ValueError(
                    f"{prefix}: need {len(expected)} original crop features, found {len(found)}; "
                    f"missing {len(missing)} (examples: {missing[:3]}). The v5 one-crop "
                    "pack is insufficient; attach original 10-crop features. No GPU work started.")
            for key, row in expected.items():
                self.features[crop_id(row["path"])] = {
                    "path": found[key], "original": row["path"], "label": row["label"]}
        except Exception:
            self.close()
            raise

    def array(self, item):
        raw = (self.archive.read(item["path"]) if self.archive else
               (self.directory / item["path"]).read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        if "sha256" in item and digest != item["sha256"]:
            raise ValueError(f"Train feature changed after validation: {item['original']}")
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        if (arr.ndim != 2 or arr.shape[1] != 512 or not len(arr) or
                arr.dtype.kind != "f" or not np.isfinite(arr).all()):
            raise ValueError(f"Invalid train shape/dtype/nonfinite values: {item['original']}")
        item.update(sha256=digest, steps=len(arr), dtype=str(arr.dtype))
        self.checked.add(item["path"])
        return arr

    def validate(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(self.features.values(), 1):
            self.array(item)
            if i % 1000 == 0:
                print(f"CPU feature validation: {i}/{len(self.features)}", flush=True)
        payload = json.dumps(list(self.features.values()), sort_keys=True, separators=(",", ":"))
        self.manifest_sha256 = hashlib.sha256(payload.encode()).hexdigest()
        (output / "validated_crop_inventory.json").write_text(payload, encoding="utf-8")
        return self.manifest_sha256

    def load_inventory(self, path):
        payload = Path(path).read_text(encoding="utf-8")
        rows = json.loads(payload)
        saved = {crop_id(item["original"]): item for item in rows}
        if len(saved) != len(rows) or set(saved) != set(self.features):
            raise ValueError("CPU validated inventory does not match crop input")
        for key, item in self.features.items():
            if item["path"] != saved[key]["path"] or item["label"] != saved[key]["label"]:
                raise ValueError("Crop path/label changed after CPU preflight")
            item.update(saved[key])
        self.manifest_sha256 = hashlib.sha256(payload.encode()).hexdigest()

    def close(self):
        if self.archive is not None:
            self.archive.close()


def recipe_batches(indices, abnormal, config, seed):
    """Native UCF: paired drop-last class loaders; XD: one shuffled all-row loader."""
    if not config.get("native_repository_recipe"):
        return list(data.balanced_batches(indices, abnormal, config["batch_size"] // 2, seed))
    rng = np.random.default_rng(seed)
    size = config["batch_size"]
    if config["dataset"] == "xd":
        order = rng.permutation(indices)
        return [order[i:i+size] for i in range(0, len(order), size)]
    if size % 2:
        raise ValueError("UCF requires an even effective batch")
    half = size // 2
    neg = rng.permutation(indices[~abnormal[indices]])
    pos = rng.permutation(indices[abnormal[indices]])
    count = min(len(neg), len(pos)) // half
    return [np.r_[neg[i*half:(i+1)*half], pos[i*half:(i+1)*half]] for i in range(count)]


def repository_video_input(array, width, device, rng=None, trace=None):
    """Match upstream process_feat default: uniform segment means, then zero pad."""
    if rng is None:
        # Development-only path retains the existing complete-video splitting.
        from train_v21 import video_input
        return video_input(array, width, device, trace=trace)
    source_steps = len(array)
    n = min(source_steps, width)
    values = np.zeros((width, 512), dtype=np.float32)
    ranges = None
    if source_steps > width:
        edges = np.linspace(0, source_steps, width + 1, dtype=np.int32)
        ranges = np.stack([edges[:-1], edges[1:]], axis=1).tolist()
        for i, (lo, hi) in enumerate(ranges):
            # Preserve source dtype in np.mean, exactly as upstream uniform_extract.
            values[i] = np.mean(array[lo:hi], axis=0)
        selected = edges[:-1].tolist()
    else:
        values[:n] = array
        selected = list(range(n))
    if trace is not None:
        trace.update(source_steps=source_steps, selected_indices=selected,
                     temporal_transform="uniform_segment_mean_then_pad",
                     pooling_ranges=ranges)
    return (torch.as_tensor(values, device=device)[None], torch.tensor([n]), n)


def audit_defaults(repo, prefix):
    """Read literal argparse defaults; fail if the pinned source disagrees."""
    path = Path(repo) / "src" / f"{prefix}_option.py"
    source = path.read_text(encoding="utf-8")
    defaults = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for kw in node.keywords:
                    if kw.arg == "default":
                        try:
                            defaults[node.args[0].value] = ast.literal_eval(kw.value)
                        except (ValueError, TypeError):
                            pass
    expected = PINNED_DEFAULTS[prefix]
    fields = {"--max-epoch": expected["epochs"], "--lr": expected["learning_rate"],
              "--batch-size": expected["batch_size"] // 2 if prefix == "ucf" else expected["batch_size"],
              "--scheduler-milestones": expected["scheduler_milestones"], "--scheduler-rate": .1,
              "--seed": 234}
    if any(defaults.get(key) != value for key, value in fields.items()):
        raise ValueError(f"Pinned option defaults differ: {prefix}: {defaults}")
    return {"option_defaults": defaults, "option_sha256": hashlib.sha256(source.encode()).hexdigest()}


def preflight_main(config):
    """Run for ALL requested datasets before allocating any CUDA model."""
    prefix, output = config["PREFIX"], Path(config["OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    data.SRC_REPO = Path(config["SRC_REPO"])
    data.checkout()
    try:
        defaults = audit_defaults(data.SRC_REPO, prefix)
        recipe = config["TRAIN_CONFIG"]
        expected = {**PINNED_DEFAULTS[prefix], "optimizer": "AdamW", "weight_decay": .01,
                    "gradient_clip": None, "amp": False, "no_decay_scalars_bias_norm": False}
        expected["microbatch_size"] = expected["batch_size"]
        if config.get("RUN_MODE") == "pilot":
            expected["epochs"] = recipe["pilot_epochs"]
        mismatch = {key: {"expected": value, "actual": recipe.get(key)}
                    for key, value in expected.items() if recipe.get(key) != value}
        if config["SEED"] != 234:
            mismatch["seed"] = {"expected": 234, "actual": config["SEED"]}
        if mismatch:
            raise ValueError(f"Native repository preset changed: {mismatch}; no silent recipe changes")
        rows, digest = data.official_csv(f"{prefix}_CLIP_rgb.csv", data.SRC_REPO)
        groups = Counter(data.video_id(r["path"]) for r in rows)
        if len(rows) != (16100 if prefix == "ucf" else 39540) or set(groups.values()) != {10}:
            raise ValueError("Expected exactly 10 crop rows per official training video")
        resolved_root = discover_train_root(config.get("TRAIN_ROOT"), config["INPUT"], prefix, rows, output)
        pack = FullCropPack(resolved_root, prefix, rows)
        try:
            inventory_hash = pack.validate(output)
            labels = np.array([r["label"] not in {"Normal", "A"} for r in rows])
            batches = recipe_batches(np.arange(len(rows)), labels, config["TRAIN_CONFIG"], config["SEED"])
            reference_updates = len(batches)
            epochs = config["TRAIN_CONFIG"]["epochs"]
            if config.get("SMOKE_ONLY"):
                batches, epochs = batches[:2], 1
            receipt = {"status": "ok", "dataset": prefix, "commit": data.COMMIT,
                       "cpu_only": True, "gpu_work_started": False,
                       "train_rows": len(rows), "unique_videos": len(groups), "crops_per_video": 10,
                       "resolved_train_root": str(resolved_root),
                       "csv_sha256": digest, "inventory_sha256": inventory_hash,
                       "updates_per_epoch": len(batches),
                       "reference_full_updates_per_epoch": reference_updates,
                       "sample_exposures_per_epoch": sum(map(len, batches)),
                       "planned_total_updates": len(batches) * epochs,
                       "resolved_recipe": config["TRAIN_CONFIG"], **defaults}
        finally:
            pack.close()
    except Exception as exc:
        receipt = {"status": "failed", "dataset": prefix, "cpu_only": True,
                   "gpu_work_started": False, "error": f"{type(exc).__name__}: {exc}"}
        (output / "preflight.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise
    (output / "preflight.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
