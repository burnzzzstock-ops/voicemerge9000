from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict


class SpeechCue(TypedDict):
    start_ms: float
    end_ms: float


class VADError(RuntimeError):
    pass


def get_speech_cues(
    vocals_wav_path: str,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 150,
) -> list[SpeechCue]:
    source = Path(vocals_wav_path).resolve()
    if not source.is_file():
        raise VADError("isolated vocal audio was not found")
    python = os.getenv("VOICEMERGE_RVC_PYTHON", sys.executable)
    command = [
        python,
        "-m",
        "engine.vad",
        "--worker",
        str(source),
        "--min-speech-ms",
        str(min_speech_duration_ms),
        "--min-silence-ms",
        str(min_silence_duration_ms),
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {
            "PATH",
            "PYTHONPATH",
            "HOME",
            "TEMP",
            "TMP",
            "TMPDIR",
            "TORCH_HOME",
            "VOICEMERGE_VAD_THRESHOLD",
            "VOICEMERGE_VAD_PAD_MS",
            "VOICEMERGE_VAD_MAX_SPEECH_SECONDS",
        }
    }
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("VOICEMERGE_VAD_TIMEOUT", "300")),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise VADError("speech detection timed out") from error
    if result.returncode:
        detail = (result.stderr or result.stdout or "Silero VAD failed").strip().splitlines()[-1]
        raise VADError(f"speech detection failed: {detail[:500]}")
    try:
        payload = json.loads(result.stdout)
        return [SpeechCue(start_ms=float(item["start_ms"]), end_ms=float(item["end_ms"])) for item in payload]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VADError("speech detection returned invalid cue data") from error


def _detect_worker(
    source: Path,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
) -> list[SpeechCue]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    torch.set_num_threads(1)
    model = load_silero_vad(onnx=True)
    waveform = read_audio(str(source), sampling_rate=16_000)
    timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=16_000,
        threshold=float(os.getenv("VOICEMERGE_VAD_THRESHOLD", "0.5")),
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=int(os.getenv("VOICEMERGE_VAD_PAD_MS", "40")),
        max_speech_duration_s=float(os.getenv("VOICEMERGE_VAD_MAX_SPEECH_SECONDS", "15")),
        return_seconds=True,
    )
    return [
        SpeechCue(
            start_ms=round(float(item["start"]) * 1000.0, 3),
            end_ms=round(float(item["end"]) * 1000.0, 3),
        )
        for item in timestamps
        if float(item["end"]) > float(item["start"])
    ]


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("source")
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=150)
    arguments = parser.parse_args()
    if not arguments.worker:
        raise SystemExit("VAD worker mode is required")
    cues = _detect_worker(Path(arguments.source), arguments.min_speech_ms, arguments.min_silence_ms)
    print(json.dumps(cues, separators=(",", ":")))


if __name__ == "__main__":
    _main()
