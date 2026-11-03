"""Synchronized, serial, feature-to-logit latency (not raw-video latency)."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


class InferenceLatency:
    def __init__(self, warmup, repeats):
        if type(warmup) is not int or warmup < 0:
            raise ValueError("LATENCY_WARMUP must be an integer >= 0")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("LATENCY_REPEATS must be an integer >= 1")
        self.warmup = warmup
        self.repeats = repeats
        self.rows = []

    def measure(self, forward, device, video_id, steps, shape, dtype):
        """Input tensors already on device; return the first measured logits."""
        device = torch.device(device)
        steps = int(steps)
        if steps < 1:
            raise ValueError("Latency requires at least one valid snippet")

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        if not self.rows:
            # Warm up once per variant/dataset process. Preserve the RNG state.
            devices = []
            if device.type == "cuda":
                devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices):
                with torch.no_grad():
                    for _ in range(self.warmup):
                        forward()
            synchronize()
        first = None
        with torch.no_grad():
            for repeat in range(self.repeats):
                synchronize()
                started = time.perf_counter()
                result = forward()
                synchronize()
                milliseconds = (time.perf_counter() - started) * 1000
                if not np.isfinite(milliseconds) or milliseconds <= 0:
                    raise ValueError("Non-positive or non-finite inference duration")
                if repeat == 0:
                    first = result
                self.rows.append({
                    "video_id": str(video_id), "repeat": repeat,
                    "snippets": steps, "input_shape": str(tuple(shape)), "input_dtype": str(dtype),
                    "latency_ms": milliseconds, "ms_per_snippet": milliseconds / steps,
                })
                del result
        return first

    def save(self, output, dataset, variant, mode, device):
        if not self.rows:
            raise ValueError("No measured inference calls")
        grouped = {}
        for row in self.rows:
            grouped.setdefault(row["video_id"], []).append(row)
        if any(len(rows) != self.repeats for rows in grouped.values()):
            raise ValueError("Expected exactly LATENCY_REPEATS measurements per video")
        per_video = np.array([
            np.median([row["latency_ms"] for row in rows]) for rows in grouped.values()
        ])
        snippets = sum(rows[0]["snippets"] for rows in grouped.values())
        total_ms = float(per_video.sum())
        summary = {
            "dataset": dataset, "variant": variant, "mode": mode,
            "scope": "device-resident padded features/mask -> logits, including window stitching "
                     "and temporal head when enabled; synchronized wall clock; serial execution",
            "excluded": "file I/O, CPU padding, H2D/D2H transfer, sigmoid/softmax scoring, "
                        "video decode, CLIP visual feature extraction, model load, metrics, training",
            "warmup_calls_first_video": self.warmup, "repeats_per_video": self.repeats,
            "warmup_policy": "first input only; new shapes may still incur first-use overhead",
            "video_count": len(grouped), "measurement_count": len(self.rows),
            "snippets": snippets,
            "aggregation": "median over repeats per video, then distribution across videos; "
                           "ms/snippet uses sum of video medians / sum of valid snippets",
            "mean_ms_per_video": float(per_video.mean()),
            "median_ms_per_video": float(np.median(per_video)),
            "p95_ms_per_video": float(np.percentile(per_video, 95)),
            "min_ms_per_video": float(per_video.min()),
            "max_ms_per_video": float(per_video.max()),
            "sum_video_medians_seconds": total_ms / 1000,
            "ms_per_snippet": total_ms / snippets,
            "snippets_per_second": snippets * 1000 / total_ms,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda"
                   else None,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cpu_threads": torch.get_num_threads(),
            "dtype_policy": "unchanged from evaluator; see recorded input dtype",
            "input_dtypes": sorted({row.get("input_dtype", "float32") for row in self.rows}),
        }
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "latency_per_video.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        (output / "latency_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Latency {dataset}/{variant}: mean={summary['mean_ms_per_video']:.3f} ms/video, "
              f"P95={summary['p95_ms_per_video']:.3f} ms/video, "
              f"{summary['ms_per_snippet']:.5f} ms/snippet", flush=True)
        return summary


def collect_latency(output):
    """Combine completed evaluator receipts, including a partial experiment."""
    output = Path(output)
    summaries = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(output.rglob("latency_summary.json"))]
    if summaries:
        with (output / "latency_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries
