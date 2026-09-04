from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    frames: int
    channels: int
    duration_seconds: float


def inspect_audio(path: str | Path) -> AudioMetadata:
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("audio file contains no decodable samples")
    return AudioMetadata(
        sample_rate=info.samplerate,
        frames=info.frames,
        channels=info.channels,
        duration_seconds=info.frames / info.samplerate,
    )


def read_audio_mono(path: str | Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if target_sample_rate and target_sample_rate != sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        mono = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = target_sample_rate
    return np.ascontiguousarray(mono), sample_rate


def write_pcm24_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.clip(audio, -1, 1), sample_rate, subtype="PCM_24", format="WAV")


def sample_bounds(start_ms: int, end_ms: int, sample_rate: int, frame_count: int) -> tuple[int, int]:
    start = min(frame_count, max(0, round(start_ms * sample_rate / 1000)))
    end = min(frame_count, max(start, round(end_ms * sample_rate / 1000)))
    return start, end


def extract_cue(master: np.ndarray, sample_rate: int, start_ms: int, end_ms: int) -> np.ndarray:
    start, end = sample_bounds(start_ms, end_ms, sample_rate, len(master))
    return np.array(master[start:end], dtype=np.float32, copy=True)


def fit_to_samples(audio: np.ndarray, sample_count: int) -> np.ndarray:
    """Correct small RVC duration drift without shifting the rest of the scene."""
    if sample_count <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(audio) == sample_count:
        return np.asarray(audio, dtype=np.float32)
    if not len(audio):
        return np.zeros(sample_count, dtype=np.float32)
    return np.interp(
        np.linspace(0, len(audio) - 1, sample_count),
        np.arange(len(audio)),
        np.asarray(audio, dtype=np.float32),
    ).astype(np.float32)


def apply_edge_fades(audio: np.ndarray, sample_rate: int, fade_ms: float = 5) -> np.ndarray:
    result = np.array(audio, dtype=np.float32, copy=True)
    width = min(len(result) // 2, max(0, round(sample_rate * fade_ms / 1000)))
    if width:
        curve = np.linspace(0, 1, width, endpoint=False, dtype=np.float32)
        result[:width] *= curve
        result[-width:] *= curve[::-1]
    return result


def assemble_speaker_stem(
    frame_count: int,
    sample_rate: int,
    cues: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    stem = np.zeros(frame_count, dtype=np.float32)
    for start_ms, end_ms, converted in cues:
        start, end = sample_bounds(start_ms, end_ms, sample_rate, frame_count)
        fitted = apply_edge_fades(fit_to_samples(converted, end - start), sample_rate)
        stem[start:end] += fitted
    return np.clip(stem, -1, 1)
