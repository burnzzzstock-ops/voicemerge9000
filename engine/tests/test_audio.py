from __future__ import annotations

import numpy as np

from engine.audio import apply_edge_fades, assemble_speaker_stem, extract_cue, fit_to_samples


def test_extract_cue_uses_sample_accurate_bounds() -> None:
    master = np.arange(1000, dtype=np.float32)
    cue = extract_cue(master, 1000, 123, 456)
    assert len(cue) == 333
    assert cue[0] == 123
    assert cue[-1] == 455


def test_fit_to_samples_corrects_duration_drift() -> None:
    source = np.linspace(-0.5, 0.5, 93, dtype=np.float32)
    fitted = fit_to_samples(source, 100)
    assert len(fitted) == 100
    assert np.isclose(fitted[0], source[0])
    assert np.isclose(fitted[-1], source[-1])


def test_edge_fades_remove_boundary_clicks() -> None:
    faded = apply_edge_fades(np.ones(1000, dtype=np.float32), 1000, fade_ms=10)
    assert faded[0] == 0
    assert faded[-1] == 0
    assert faded[20] == 1


def test_assemble_stem_preserves_silence_and_timeline_length() -> None:
    cue = np.full(100, 0.25, dtype=np.float32)
    stem = assemble_speaker_stem(1000, 1000, [(200, 300, cue), (600, 700, cue)])
    assert len(stem) == 1000
    assert np.all(stem[:200] == 0)
    assert np.any(stem[200:300] != 0)
    assert np.all(stem[300:600] == 0)
    assert np.any(stem[600:700] != 0)
    assert np.all(stem[700:] == 0)
