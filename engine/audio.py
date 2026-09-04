from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
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


def read_audio(path: str | Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if target_sample_rate and target_sample_rate != sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            sample_rate // divisor,
            axis=0,
        ).astype(np.float32)
        sample_rate = target_sample_rate
    return np.ascontiguousarray(audio), sample_rate


def read_audio_mono(path: str | Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sample_rate = read_audio(path, target_sample_rate)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    return np.ascontiguousarray(mono), sample_rate


def write_pcm24_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.clip(audio, -1, 1), sample_rate, subtype="PCM_24", format="WAV")


def write_float_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Persist an intermediate stem without quantizing or clipping it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT", format="WAV")


def sample_bounds(
    start_ms: float,
    end_ms: float,
    sample_rate: int,
    frame_count: int,
) -> tuple[int, int]:
    start = min(frame_count, max(0, round(start_ms * sample_rate / 1000.0)))
    end = min(frame_count, max(start, round(end_ms * sample_rate / 1000.0)))
    return start, end


def extract_cue_from_array(
    audio: np.ndarray,
    start_ms: float,
    end_ms: float,
    sample_rate: int,
) -> np.ndarray:
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")
    start, end = sample_bounds(start_ms, end_ms, sample_rate, len(audio))
    return np.array(audio[start:end], dtype=np.float32, copy=True)


def extract_cue(master: np.ndarray, sample_rate: int, start_ms: float, end_ms: float) -> np.ndarray:
    """Backward-compatible name for callers that extract a reviewed cue."""
    return extract_cue_from_array(master, start_ms, end_ms, sample_rate)


def _empty_like_length(audio: np.ndarray, length: int) -> np.ndarray:
    return np.zeros((length, *audio.shape[1:]), dtype=np.float32)


def pad_or_trim_clip(clip: np.ndarray, target_length: int, sample_rate: int = 48_000) -> np.ndarray:
    """Reconcile a cue without resampling, pitch shifting, or stretching it."""
    source = np.asarray(clip, dtype=np.float32)
    if source.ndim not in {1, 2}:
        raise ValueError("audio clips must be mono or sample-major multichannel arrays")
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if len(source) == target_length:
        return np.array(source, copy=True)
    if len(source) < target_length:
        result = _empty_like_length(source, target_length)
        result[: len(source)] = source
        return result
    if target_length == 0:
        return _empty_like_length(source, 0)

    result = np.array(source[:target_length], copy=True)
    fade_samples = min(target_length, max(1, round(sample_rate * 0.005)))
    fade = np.linspace(1.0, 0.0, fade_samples, endpoint=True, dtype=np.float32)
    if result.ndim == 2:
        fade = fade[:, None]
    result[-fade_samples:] *= fade
    return result


def apply_edge_fades(audio: np.ndarray, sample_rate: int, fade_ms: float = 5) -> np.ndarray:
    result = np.array(audio, dtype=np.float32, copy=True)
    width = min(len(result) // 2, max(0, round(sample_rate * fade_ms / 1000)))
    if width:
        curve = np.linspace(0, 1, width, endpoint=False, dtype=np.float32)
        if result.ndim == 2:
            curve = curve[:, None]
        result[:width] *= curve
        result[-width:] *= curve[::-1]
    return result


def assemble_speaker_stem(
    converted_cues: list[tuple[float, float, np.ndarray]],
    total_samples: int,
    sample_rate: int,
) -> np.ndarray:
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    channel_shape: tuple[int, ...] = ()
    if converted_cues:
        first = np.asarray(converted_cues[0][2])
        if first.ndim not in {1, 2}:
            raise ValueError("converted cues must be mono or sample-major multichannel arrays")
        channel_shape = first.shape[1:]
    stem = np.zeros((total_samples, *channel_shape), dtype=np.float32)
    for start_ms, end_ms, converted in converted_cues:
        start, end = sample_bounds(start_ms, end_ms, sample_rate, total_samples)
        if end <= start:
            continue
        fitted = pad_or_trim_clip(converted, end - start, sample_rate)
        fitted = apply_edge_fades(fitted, sample_rate)
        if fitted.shape[1:] != channel_shape:
            raise ValueError("all converted cues must use the same channel layout")
        stem[start:end] += fitted
    return stem


def _measure_loudness(audio: np.ndarray, sample_rate: int) -> float | None:
    if len(audio) < round(sample_rate * 0.4) or not np.any(np.abs(audio) > 1e-7):
        return None
    try:
        measured = float(pyln.Meter(sample_rate).integrated_loudness(audio))
    except (FloatingPointError, OverflowError, ValueError):
        return None
    return measured if np.isfinite(measured) else None


def _true_peak(audio: np.ndarray, sample_rate: int) -> float:
    if not len(audio):
        return 0.0
    peak = 0.0
    chunk_samples = max(1, sample_rate * 10)
    overlap_samples = 64
    for start in range(0, len(audio), chunk_samples):
        left = max(0, start - overlap_samples)
        right = min(len(audio), start + chunk_samples + overlap_samples)
        oversampled = resample_poly(audio[left:right], 4, 1, axis=0)
        peak = max(peak, float(np.max(np.abs(oversampled), initial=0.0)))
    return peak


def peak_limit_and_normalize(
    vocal_stem: np.ndarray,
    target_lufs: float = -18.0,
    peak_limit: float = -1.0,
    sample_rate: int = 48_000,
) -> np.ndarray:
    """Apply bounded loudness gain and an oversampled true-peak ceiling."""
    audio = np.asarray(vocal_stem, dtype=np.float32)
    if audio.ndim not in {1, 2}:
        raise ValueError("audio must be mono or sample-major multichannel")
    if not len(audio):
        return np.array(audio, copy=True)
    result = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)
    measured = _measure_loudness(result, sample_rate)
    if measured is not None:
        gain_db = float(np.clip(target_lufs - measured, -24.0, 18.0))
        result *= np.float32(10 ** (gain_db / 20.0))

    ceiling = float(10 ** (peak_limit / 20.0))
    true_peak = _true_peak(result, sample_rate)
    if true_peak > ceiling and true_peak > 0:
        result *= np.float32(ceiling / true_peak)
    return np.clip(result, -ceiling, ceiling).astype(np.float32, copy=False)


def mix_background_and_stems(
    background: np.ndarray,
    speaker_stems: list[np.ndarray],
    sample_rate: int,
    target_lufs: float = -16.0,
    peak_limit: float = -1.0,
) -> np.ndarray:
    bed = np.asarray(background, dtype=np.float32)
    if bed.ndim != 2 or bed.shape[1] not in {1, 2}:
        raise ValueError("background must be a sample-major mono or stereo array")
    master = np.array(bed, copy=True)
    for stem in speaker_stems:
        fitted = pad_or_trim_clip(np.asarray(stem, dtype=np.float32), len(master), sample_rate)
        if fitted.ndim == 1:
            fitted = fitted[:, None]
        if fitted.shape[1] == 1 and master.shape[1] == 2:
            fitted = np.repeat(fitted, 2, axis=1)
        if fitted.shape[1] != master.shape[1]:
            raise ValueError("speaker stem channels do not match the background")
        master += fitted
    return peak_limit_and_normalize(master, target_lufs, peak_limit, sample_rate)
