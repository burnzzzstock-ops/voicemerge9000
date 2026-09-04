from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .model_store import ModelArtifact
from .schemas import RVCParams


class RVCInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RVCSettings:
    backend: str
    root: Path
    python: str
    timeout_seconds: int

    @classmethod
    def from_environment(cls) -> "RVCSettings":
        return cls(
            backend=os.getenv("VOICEMERGE_RVC_BACKEND", "official-rvc"),
            root=Path(os.getenv("VOICEMERGE_RVC_ROOT", "/opt/rvc")),
            python=os.getenv("VOICEMERGE_RVC_PYTHON", sys.executable),
            timeout_seconds=int(os.getenv("VOICEMERGE_RVC_TIMEOUT", "900")),
        )


class RVCEngine:
    def __init__(self, settings: RVCSettings | None = None) -> None:
        self.settings = settings or RVCSettings.from_environment()
        if self.settings.backend == "copy" and os.getenv("VOICEMERGE_ENV", "production") not in {"test", "development"}:
            raise RVCInferenceError("the copy backend is permitted only in development or tests")

    @property
    def backend(self) -> str:
        return self.settings.backend

    def device(self) -> str:
        if self.settings.backend == "copy":
            return "development"
        try:
            result = subprocess.run(
                [self.settings.python, "-c", "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            return result.stdout.strip() or "cpu"
        except (OSError, subprocess.SubprocessError):
            return "cpu"

    def ready(self) -> None:
        if self.settings.backend == "copy":
            return
        if self.settings.backend != "official-rvc":
            raise RVCInferenceError(f"unsupported RVC backend: {self.settings.backend}")
        if not (self.settings.root / "infer" / "cli.py").is_file():
            raise RVCInferenceError("official RVC CLI was not found; check VOICEMERGE_RVC_ROOT")

    def convert_batch(
        self,
        sources: list[Path],
        output_directory: Path,
        model: ModelArtifact,
        params: RVCParams,
    ) -> dict[Path, Path]:
        if not sources:
            return {}
        input_directory = sources[0].parent
        if any(source.parent != input_directory for source in sources):
            raise RVCInferenceError("batch inputs must share one directory")
        output_directory.mkdir(parents=True, exist_ok=True)
        if self.settings.backend == "copy":
            outputs = {}
            for source in sources:
                destination = output_directory / source.name
                shutil.copyfile(source, destination)
                outputs[source] = destination
            return outputs
        self.ready()
        index_alias: Path | None = None
        if model.index:
            # The official CLI rewrites any supplied filename containing
            # "trained" to "added" before checking that it exists. Community
            # archives often include only the trained-named file, so expose the
            # validated index through a stable alias that the CLI will accept.
            index_alias = output_directory / ".voicemerge.index"
            index_alias.unlink(missing_ok=True)
            try:
                os.link(model.index, index_alias)
            except OSError:
                try:
                    index_alias.symlink_to(model.index)
                except OSError:
                    shutil.copyfile(model.index, index_alias)
        command = [
            self.settings.python,
            "-m", "infer.cli",
            "--model", str(model.checkpoint),
            "--input", str(input_directory),
            "--output", str(output_directory),
            "--pitch", str(params.pitch),
            "--f0-method", params.f0_method,
            "--index-rate", str(params.index_rate if index_alias else 0),
            "--rms-mix-rate", str(params.rms_mix_rate),
            "--protect", str(params.protect),
            "--resample-sr", "0",
            "--format", "wav",
            "--overwrite",
        ]
        if index_alias:
            command.extend(["--index", str(index_alias)])
        environment = {
            key: value for key, value in os.environ.items()
            if key in {"PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "HOME", "TEMP", "TMP", "TMPDIR"}
        }
        environment["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            result = subprocess.run(
                command,
                cwd=self.settings.root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RVCInferenceError("RVC conversion timed out") from error
        finally:
            if index_alias:
                index_alias.unlink(missing_ok=True)
            gc.collect()
        outputs = {source: output_directory / source.name for source in sources}
        if result.returncode or any(not destination.is_file() for destination in outputs.values()):
            detail = (result.stderr or result.stdout or "RVC produced no output").strip().splitlines()[-1]
            raise RVCInferenceError(f"RVC conversion failed: {detail[:500]}")
        return outputs

    def convert(self, source: Path, destination: Path, model: ModelArtifact, params: RVCParams) -> None:
        temporary_output = destination.parent / f".{destination.stem}-batch"
        outputs = self.convert_batch([source], temporary_output, model, params)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(outputs[source], destination)
        shutil.rmtree(temporary_output, ignore_errors=True)
