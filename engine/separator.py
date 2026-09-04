from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path


class SeparationError(RuntimeError):
    pass


def _worker_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "HOME",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCH_HOME",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
        "VOICEMERGE_DEMUCS_MODEL",
        "VOICEMERGE_DEMUCS_SEGMENT",
        "VOICEMERGE_DEMUCS_OVERLAP",
        "VOICEMERGE_DEMUCS_PRECISION",
        "VOICEMERGE_MAX_GPU_GB",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def separate_vocals(audio_path: str, output_dir: str) -> tuple[str, str]:
    source = Path(audio_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise SeparationError("source audio was not found")
    destination.mkdir(parents=True, exist_ok=True)
    vocals = destination / "vocals.wav"
    background = destination / "background.wav"
    timeout = int(os.getenv("VOICEMERGE_SEPARATION_TIMEOUT", "2400"))
    python = os.getenv("VOICEMERGE_RVC_PYTHON", sys.executable)
    command = [
        python,
        "-m",
        "engine.separator",
        "--worker",
        str(source),
        str(destination),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            env=_worker_environment(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise SeparationError("vocal separation timed out") from error
    if result.returncode or not vocals.is_file() or not background.is_file():
        detail = (result.stderr or result.stdout or "Demucs produced no stems").strip().splitlines()[-1]
        raise SeparationError(f"vocal separation failed: {detail[:500]}")
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    return str(vocals), str(background)


def _separate_worker(source: Path, output_dir: Path) -> None:
    import soundfile as sf
    import torch
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, convert_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.getenv("VOICEMERGE_DEMUCS_MODEL", "htdemucs")
    segment = float(os.getenv("VOICEMERGE_DEMUCS_SEGMENT", "7.0"))
    overlap = float(os.getenv("VOICEMERGE_DEMUCS_OVERLAP", "0.1"))
    precision = os.getenv("VOICEMERGE_DEMUCS_PRECISION", "fp16").lower()
    model = None
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = pretrained.get_model(model_name)
        model.eval()
        mix = AudioFile(source).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
        mix = convert_audio(mix, model.samplerate, model.samplerate, model.audio_channels)
        reference = mix.mean(0)
        mean = reference.mean()
        scale = reference.std() + 1e-8
        normalized = (mix - mean) / scale
        use_fp16 = device == "cuda" and precision == "fp16"
        model.to(device)
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_fp16
            else nullcontext()
        )
        with torch.inference_mode(), precision_context:
            separated = apply_model(
                model,
                normalized[None],
                shifts=0,
                split=True,
                overlap=overlap,
                segment=segment,
                device=device,
                num_workers=0,
                progress=False,
            )[0]
        separated = separated.float().cpu() * scale.float().cpu() + mean.float().cpu()
        source_names = list(model.sources)
        vocals_index = source_names.index("vocals")
        vocals = separated[vocals_index]
        background = separated[[index for index, name in enumerate(source_names) if name != "vocals"]].sum(0)
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(output_dir / "vocals.wav", vocals.transpose(0, 1).numpy(), model.samplerate, subtype="FLOAT")
        sf.write(output_dir / "background.wav", background.transpose(0, 1).numpy(), model.samplerate, subtype="FLOAT")
        if device == "cuda":
            peak = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) / 1024**3
            print(f"demucs_peak_vram_gb={peak:.3f}", flush=True)
            if peak > float(os.getenv("VOICEMERGE_MAX_GPU_GB", "6.5")):
                raise SeparationError(f"Demucs exceeded the configured GPU budget ({peak:.2f} GB)")
    finally:
        del model
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("source")
    parser.add_argument("output_dir")
    arguments = parser.parse_args()
    if not arguments.worker:
        raise SystemExit("separator worker mode is required")
    output = Path(arguments.output_dir)
    if output.exists():
        for name in ("vocals.wav", "background.wav"):
            (output / name).unlink(missing_ok=True)
    _separate_worker(Path(arguments.source), output)


if __name__ == "__main__":
    _main()
