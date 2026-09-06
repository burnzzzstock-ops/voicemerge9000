"""Bound PyTorch allocations; total CUDA/device usage must also be measured."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


def configure_budget(torch: Any) -> dict[str, float | str]:
    budget = float(os.getenv("VOICEMERGE_MAX_GPU_GB", "6.5"))
    if not math.isfinite(budget) or not 0.75 <= budget <= 6.5:
        raise ValueError("GPU budget must be between 0.75 and 6.5 GiB")
    metrics: dict[str, float | str] = {"budget_gib": budget, "device": "cpu"}
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        allocator = min((budget - 0.5) * 1024**3, total * 0.9)
        torch.cuda.set_per_process_memory_fraction(allocator / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        metrics.update(device="cuda", allocator_budget_gib=allocator / 1024**3)
    return metrics


def save_metrics(torch: Any, path: Path, metrics: dict[str, float | str]) -> None:
    if torch.cuda.is_available():
        metrics.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"gpu_metrics": metrics}), flush=True)
