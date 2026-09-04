from __future__ import annotations

import numpy as np

from engine.audio import (
    assemble_speaker_stem,
    extract_cue_from_array,
    mix_background_and_stems,
    pad_or_trim_clip,
    peak_limit_and_normalize,
)


def test_timing_and_no_interp() -> None:
    sample_rate = 48_000
    source = np.zeros(sample_rate * 10, dtype=np.float32)
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source[sample_rate * 2 : sample_rate * 3] = np.sin(2 * np.pi * 440 * time)

    cue = extract_cue_from_array(source, 2_000, 3_000, sample_rate)
    short = pad_or_trim_clip(cue[:-200], sample_rate, sample_rate)
    long = pad_or_trim_clip(np.pad(cue, (0, 200)), sample_rate, sample_rate)
    short_stem = assemble_speaker_stem([(2_000, 3_000, short)], len(source), sample_rate)
    long_stem = assemble_speaker_stem([(2_000, 3_000, long)], len(source), sample_rate)

    assert len(short_stem) == 480_000
    assert len(long_stem) == 480_000
    assert np.flatnonzero(np.abs(short_stem) > 1e-4)[0] in range(96_000, 96_010)
    assert np.flatnonzero(np.abs(long_stem) > 1e-4)[0] in range(96_000, 96_010)


def test_zero_clipping() -> None:
    audio = np.full((48_000, 2), 2.0, dtype=np.float32)
    limited = peak_limit_and_normalize(audio, target_lufs=-18.0, peak_limit=-1.0)

    assert float(np.max(np.abs(limited))) <= 10 ** (-1.0 / 20.0)


def test_master_mix_keeps_exact_background_shape_and_peak() -> None:
    background = np.full((48_000, 2), 0.2, dtype=np.float32)
    short = np.ones(47_800, dtype=np.float32)
    long = np.ones(48_200, dtype=np.float32)

    mixed = mix_background_and_stems(background, [short, long], 48_000)

    assert mixed.shape == background.shape
    assert float(np.max(np.abs(mixed))) <= 10 ** (-1.0 / 20.0)
