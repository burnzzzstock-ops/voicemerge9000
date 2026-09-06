"""Run the official CLI with module import semantics after reserving GPU headroom."""
from __future__ import annotations

import argparse
import gc
import os
import runpy
import sys
import time
from pathlib import Path

from .gpu_budget import configure_budget, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    import torch
    started = time.monotonic()
    metrics = configure_budget(torch)
    metrics["stage"] = "rvc"
    metrics["status"] = "failed"
    try:
        root = args.root.resolve()
        sys.path.insert(0, str(root))
        os.chdir(root)
        sys.argv = ["infer.cli", *remaining]
        try:
            runpy.run_module("infer.cli", run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        metrics["status"] = "completed"
    finally:
        metrics["duration_seconds"] = time.monotonic() - started
        save_metrics(torch, args.metrics, metrics)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
