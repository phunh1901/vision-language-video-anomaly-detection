"""Reuse completed full-run datasets after postprocessing failures; never resume training."""

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


class RecoveryInput:
    def __init__(self, source):
        self.path = Path(source).expanduser().resolve()
        if self.path.is_dir():
            self.archive = None
            self.names = [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]
        elif self.path.is_file() and self.path.suffix.lower() == ".zip":
            self.archive = ZipFile(self.path)
            self.names = [info.filename for info in self.archive.infolist() if not info.is_dir()]
        else:
            raise FileNotFoundError(f"RECOVERY_FROM must be the previous run directory or result ZIP: {self.path}")
        if "experiment.json" not in self.names or len(self.names) != len(set(self.names)):
            self.close()
            raise ValueError("Recovery input must contain one run with experiment.json at its root")
        for name in self.names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name:
                self.close()
                raise ValueError("Unsafe recovery archive/path member")

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self, name):
        if name not in self.names:
            raise FileNotFoundError(f"Recovery artifact missing: {name}")
        if self.archive is not None:
            return self.archive.read(name)
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError("Recovery file resolves outside source run")
        return path.read_bytes()

    def metadata(self, name):
        return json.loads(self.read(name).decode("utf-8"))

    def reuse(self, prefix, output, sources, train_config, geometry_config, seed, checkpoint_sha, manifest_sha, treatment):
        """Return copied report paths only after provenance/recipe/checkpoint validation."""
        training_name = f"{prefix}/train/training.json"
        if training_name not in self.names:
            return None
        experiment = self.metadata("experiment.json")
        if experiment.get("mode") != "full" or experiment.get("seed") != seed:
            raise ValueError("Cannot reuse a pilot/smoke run or another seed as a full benchmark")
        if experiment.get("geometry_config") != geometry_config:
            raise ValueError("Recovery geometry configuration differs")
        mutable = {"v21_error_analysis.py", "orchestrator_v21.py", "v21_recovery.py"}
        for name, source in sources.items():
            actual_sha = hashlib.sha256(source.encode()).hexdigest()
            previous_sha = experiment["source_sha256"].get(name)
            if name == "v21_diagnostics.py" and previous_sha != actual_sha:
                previous = self.read("runtime_scripts/v21_diagnostics.py").decode("utf-8")
                if hashlib.sha256(previous.encode()).hexdigest() != previous_sha:
                    raise ValueError("Recovery diagnostic source does not match its manifest")

                def training_contract(text):
                    tree = ast.parse(text)
                    tree.body = [node for node in tree.body if not
                                 (isinstance(node, ast.ClassDef) and node.name == "EvaluationTrace")]
                    return ast.dump(tree, include_attributes=False)

                if training_contract(previous) != training_contract(source):
                    raise ValueError("Recovery diagnostics changed training behavior; cannot reuse")
                continue  # Only the observational inference trace class changed.
            if name not in mutable and previous_sha != actual_sha:
                raise ValueError(f"Recovery model/data/training source differs: {name}")
        training = self.metadata(training_name)
        record = training["variants"][treatment]
        def canonical(value):
            return json.loads(json.dumps(value))
        if canonical(record["config"]) != canonical(train_config):
            raise ValueError("Recovery training recipe differs; no checkpoint or metrics reused")
        if (training.get("feature_manifest_sha256") != manifest_sha or
                training.get("initial_v0_sha256") != checkpoint_sha or
                record.get("initial_v0_sha256") != checkpoint_sha):
            raise ValueError("Recovery input/checkpoint provenance differs")
        checkpoint_name = f"{prefix}/train/{record['checkpoint_file']}"
        if hashlib.sha256(self.read(checkpoint_name)).hexdigest() != record["checkpoint_sha256"]:
            raise ValueError("Recovery trained checkpoint SHA-256 mismatch")
        paths = {}
        for variant in ("v0", treatment):
            matches = [n for n in self.names if n.startswith(f"{prefix}/{variant}/") and n.endswith("/run.json")]
            if len(matches) != 1:
                raise ValueError(f"Recovery has completed training but lacks one full evaluator report: {prefix}/{variant}")
            name = matches[0]
            report = self.metadata(name)
            expected_sha = checkpoint_sha if variant == "v0" else record["checkpoint_sha256"]
            if (report.get("evaluated_checkpoint_sha256") != expected_sha or
                    report.get("initial_v0_checkpoint_sha256") != checkpoint_sha or
                    report.get("run_kind") != "full_checkpoint_inference" or not report.get("metrics")):
                raise ValueError(f"Recovery evaluator checkpoint/mode mismatch: {prefix}/{variant}")
            folder = PurePosixPath(name).parent
            for required in ("predictions.npz", "diagnostics/diagnostic_tensors.npz"):
                if str(folder / required) not in self.names:
                    raise ValueError(f"Recovery evaluator artifact missing: {prefix}/{variant}/{required}")
            paths[variant] = Path(output) / name
        copied = []
        for name in self.names:
            if not name.startswith(prefix + "/"):
                continue
            relative = PurePosixPath(name)
            if (relative.suffix not in {".json", ".jsonl", ".csv", ".log", ".md", ".npz"}
                    and relative.name != "best.pth"):
                continue
            if len(relative.parts) > 1 and relative.parts[1] == "error_analysis":
                continue  # Rebuild with the corrected analyzer.
            target = Path(output) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = self.read(name)
            target.write_bytes(payload)
            copied.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest()})
        receipt = {"source_run": self.path.name, "dataset": prefix, "source_experiment": experiment,
                   "source_experiment_sha256": hashlib.sha256(self.read("experiment.json")).hexdigest(),
                   "new_training": False, "new_inference": False, "postprocessing_rerun": True,
                   "files": copied, "latency_scope": "reused original V0/treatment pair; not measured in this session"}
        (Path(output) / prefix / "recovery_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return paths, Path(output) / training_name
