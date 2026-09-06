from __future__ import annotations

import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from scipy.signal import resample_poly

from test_api import HEADERS, api_module as api, fake_separate_vocals, wait_for_completion, wav_bytes
from engine.audio import mix_background_and_stems, peak_limit_and_normalize, write_pcm24_wav
from engine.gpu_budget import configure_budget
from engine.vad import _detect_worker


@pytest.fixture
def analyzed(monkeypatch):
    monkeypatch.setattr(api, "separate_vocals", fake_separate_vocals)
    monkeypatch.setattr(api, "get_speech_cues", lambda _: [{"start_ms": 100.0625, "end_ms": 300.0625}])
    with TestClient(api.app) as client:
        data = client.post("/api/v1/analyze", headers=HEADERS, files={"audio_file": ("test.wav", wav_bytes(), "audio/wav")}).json()
        yield client, data
        record = api.jobs.get(data["job_id"])
        if record and record.status not in {"queued", "processing"}:
            client.delete(f"/api/v1/jobs/{data['job_id']}", headers=HEADERS)


def timeline(job_id):
    return {"job_id": job_id, "timeline": [
        {"speaker_id": "speaker-1", "model_id": "development", "cues": [{"cue_id": "a", "start_ms": 100.0625, "end_ms": 200.0625}]},
        {"speaker_id": "speaker-2", "model_id": "development", "cues": [{"cue_id": "b", "start_ms": 200.0625, "end_ms": 300.0625}]},
    ]}


def test_vad_keeps_sample_offsets(monkeypatch):
    captured = {}
    def timestamps(*args, **kwargs):
        captured.update(kwargs)
        return [{"start": 1601, "end": 4801}]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    monkeypatch.setitem(sys.modules, "silero_vad", SimpleNamespace(get_speech_timestamps=timestamps, load_silero_vad=lambda **_: object(), read_audio=lambda *_, **__: object()))
    cues = _detect_worker(Path("unused.wav"), 250, 150)
    assert captured["return_seconds"] is False
    assert cues == [{"start_ms": 100.0625, "end_ms": 300.0625}]


def test_api_preserves_vad_precision_and_safe_edits(analyzed):
    client, data = analyzed
    assert data["cues"][0]["start_ms"] == 100.0625
    record = api.get_job(data["job_id"])
    from engine.schemas import JobSubmission
    valid = timeline(data["job_id"])["timeline"][:1]
    valid[0]["cues"][0]["start_ms"] = 50.0625
    api._validate_timeline(record, JobSubmission(timeline=valid))
    valid[0]["cues"][0].update(start_ms=320, end_ms=400)
    with pytest.raises(ValueError, match="outside"):
        api._validate_timeline(record, JobSubmission(timeline=valid))


def test_duplicate_conversion_reserved_once(analyzed, monkeypatch):
    client, data = analyzed
    submitted = []
    monkeypatch.setattr(api, "executor", SimpleNamespace(submit=lambda *args: submitted.append(args)))
    gate = threading.Barrier(2)
    def request():
        gate.wait()
        return client.post("/api/v1/convert", headers=HEADERS, json=timeline(data["job_id"])).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request) for _ in range(2)]
        assert sorted(f.result() for f in futures) == [202, 409]
    assert len(submitted) == 1
    api.get_job(data["job_id"]).status = "failed"


@pytest.mark.parametrize("total_samples", [48002, 48003, 48004, 48011, 48012])
def test_final_cue_accepts_browser_roundoff_not_an_extra_sample(total_samples):
    from engine.schemas import JobSubmission, SpeechCueResult
    duration_ms = total_samples * 1000.0 / 48000
    record = SimpleNamespace(total_samples=total_samples, sample_rate=48000,
                             detected_cues=[SpeechCueResult(cue_id="tail", start_ms=500, end_ms=duration_ms)])
    cue = {"cue_id": "tail", "start_ms": 500, "end_ms": (duration_ms / 1000) * 1000}
    def submission():
        return JobSubmission(timeline=[{"speaker_id": "speaker-1", "model_id": "development", "cues": [cue]}])
    api._validate_timeline(record, submission())
    cue["end_ms"] = duration_ms + 1000.0 / 48000
    with pytest.raises(ValueError, match="scene duration"):
        api._validate_timeline(record, submission())


def test_human_padding_limits_accept_roundoff_not_an_extra_sample():
    from engine.schemas import CueSegment, JobSubmission, SpeechCueResult
    record = SimpleNamespace(total_samples=96000, sample_rate=48000,
                             detected_cues=[SpeechCueResult(cue_id="speech", start_ms=650.0625, end_ms=850.0625)])
    cue = {"cue_id": "speech", "start_ms": (500.0625 / 1000) * 1000,
           "end_ms": (1000.0625 / 1000) * 1000}
    def submission():
        return JobSubmission(timeline=[{"speaker_id": "speaker-1", "model_id": "development", "cues": [cue]}])
    api._validate_timeline(record, submission())
    cue["end_ms"] += 1000.0 / 48000
    with pytest.raises(ValueError, match="outside"):
        api._validate_timeline(record, submission())
    CueSegment(cue_id="forty-ms", start_ms=100.1, end_ms=140.1)
    with pytest.raises(ValueError, match="40 ms"):
        CueSegment(cue_id="short", start_ms=100.1, end_ms=140.1 - 1000.0 / 48000)


def test_partial_failure_retry_and_immutable_master(analyzed, monkeypatch):
    client, data = analyzed
    original = api.engine.convert_batch
    calls = []
    fail = True
    def convert(sources, output, model, params):
        speaker = sources[0].parent.parent.name
        calls.append(speaker)
        if fail and speaker == "speaker-2":
            raise RuntimeError("test model failure")
        return original(sources, output, model, params)
    monkeypatch.setattr(api.engine, "convert_batch", convert)
    payload = timeline(data["job_id"])
    assert client.post("/api/v1/convert", headers=HEADERS, json=payload).status_code == 202
    first = wait_for_completion(client, data["job_id"])
    assert first["status"] == "failed" and "master" not in first["tracks"]
    old_voice = first["tracks"]["speaker-1"]
    fail = False
    client.post("/api/v1/convert", headers=HEADERS, json=payload)
    second = wait_for_completion(client, data["job_id"])
    assert second["status"] == "completed"
    assert calls == ["speaker-1", "speaker-2", "speaker-2"]
    assert client.get(old_voice, headers=HEADERS).status_code == 200
    old_master = second["tracks"]["master"]
    old_bytes = client.get(old_master, headers=HEADERS).content
    old_voice_bytes = client.get(old_voice, headers=HEADERS).content
    payload["timeline"][0]["gain"] = 0.8
    client.post("/api/v1/convert", headers=HEADERS, json=payload)
    third = wait_for_completion(client, data["job_id"])
    assert third["status"] == "completed"
    assert calls == ["speaker-1", "speaker-2", "speaker-2"]
    assert client.get(third["tracks"]["speaker-1"], headers=HEADERS).content == old_voice_bytes
    assert client.get(old_master, headers=HEADERS).content == old_bytes
    new_bytes = client.get(third["tracks"]["master"], headers=HEADERS).content
    assert new_bytes != old_bytes
    def balance(audio_bytes):
        audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
        assert sr == 48000 and len(audio) == 24000
        assert np.max(np.abs(resample_poly(audio, 4, 1, axis=0))) <= 10 ** (-1 / 20) + 2e-6
        a = audio[round(.11 * sr):round(.19 * sr)]
        b = audio[round(.21 * sr):round(.29 * sr)]
        return np.sqrt(np.mean(a**2) / np.mean(b**2))
    assert balance(new_bytes) / balance(old_bytes) == pytest.approx(.8, abs=2e-5)


def test_fresh_and_cached_nondefault_gain_match(analyzed, monkeypatch):
    client, data = analyzed
    calls = []
    original = api.engine.convert_batch
    def convert(sources, output, model, params):
        calls.append(sources[0].parent.parent.name)
        return original(sources, output, model, params)
    monkeypatch.setattr(api.engine, "convert_batch", convert)
    payload = timeline(data["job_id"])
    payload["timeline"][1]["gain"] = 1.75
    def render():
        assert client.post("/api/v1/convert", headers=HEADERS, json=payload).status_code == 202
        result = wait_for_completion(client, data["job_id"])
        assert result["status"] == "completed"
        return client.get(result["tracks"]["master"], headers=HEADERS).content
    fresh = render()
    assert render() == fresh
    assert calls == ["speaker-1", "speaker-2"]
    payload["timeline"][0]["model_id"] = "changed-test-model"
    render()
    assert calls == ["speaker-1", "speaker-2", "speaker-1"]
    payload["timeline"][0]["cues"][0]["start_ms"] += 1
    render()
    assert calls[-2:] == ["speaker-1", "speaker-1"]
    assert len(calls) == 4
    payload["params"] = {"pitch": 1}
    render()
    assert calls[-2:] == ["speaker-1", "speaker-2"]
    assert len(calls) == 6


def test_analysis_is_pollable_before_separation_finishes(monkeypatch):
    started, release = threading.Event(), threading.Event()
    def separate(source, target):
        started.set()
        assert release.wait(5)
        return fake_separate_vocals(source, target)
    monkeypatch.setattr(api, "separate_vocals", separate)
    monkeypatch.setattr(api, "get_speech_cues", lambda _: [])
    with TestClient(api.app) as client:
        response = client.post("/api/v1/analyze?background=true", headers=HEADERS, files={"audio_file": ("test.wav", wav_bytes(), "audio/wav")})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        try:
            assert started.wait(2)
            assert client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()["stage"] == "separating_vocals"
        finally:
            release.set()
        for _ in range(100):
            result = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()
            if result["status"] == "analyzed":
                break
            time.sleep(.02)
        assert result["analysis"]["cues"] == []
        client.delete(f"/api/v1/jobs/{job_id}", headers=HEADERS)


def test_true_peak_after_pcm24_export(tmp_path):
    sr = 48000
    t = np.arange(sr) / sr
    signal = (np.sin(2 * np.pi * 11000 * t + .8) * 2).astype(np.float32)
    output = peak_limit_and_normalize(signal, target_lufs=-8)
    path = tmp_path / "master.wav"
    write_pcm24_wav(path, output, sr)
    exported, _ = sf.read(path)
    assert np.max(np.abs(resample_poly(exported, 4, 1))) <= 10 ** (-1 / 20) + 2e-6


def test_stereo_side_survives_common_master_gain():
    sr = 48000
    t = np.arange(sr) / sr
    bed = np.column_stack([.1 * np.sin(2*np.pi*220*t), .07 * np.sin(2*np.pi*330*t)]).astype(np.float32)
    vocal = (.05 * np.sin(2*np.pi*440*t)).astype(np.float32)
    output = mix_background_and_stems(bed, [vocal], sr)
    side = bed[:, 0] - bed[:, 1]
    result_side = output[:, 0] - output[:, 1]
    gain = np.dot(side, result_side) / np.dot(side, side)
    assert gain > 0
    np.testing.assert_allclose(result_side, side * gain, atol=2e-7)


def test_gpu_budget_cpu_and_validation(monkeypatch):
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert configure_budget(fake)["device"] == "cpu"
    monkeypatch.setenv("VOICEMERGE_MAX_GPU_GB", "nan")
    with pytest.raises(ValueError):
        configure_budget(fake)


def test_running_job_cannot_be_deleted_or_expired(analyzed, monkeypatch):
    client, data = analyzed
    monkeypatch.setattr(api, "executor", SimpleNamespace(submit=lambda *_: None))
    assert client.post("/api/v1/convert", headers=HEADERS, json=timeline(data["job_id"])).status_code == 202
    record = api.get_job(data["job_id"])
    record.created_at = 0
    api.purge_expired_jobs()
    assert client.delete(f"/api/v1/jobs/{record.job_id}", headers=HEADERS).status_code == 409
    assert record.directory.is_dir()
    record.status = "failed"


def test_attempt_setup_failure_is_recoverable(analyzed, monkeypatch):
    client, data = analyzed
    original = Path.mkdir
    def fail_work(path, *args, **kwargs):
        if path.name == "work":
            raise OSError("test disk full")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", fail_work)
    client.post("/api/v1/convert", headers=HEADERS, json=timeline(data["job_id"]))
    result = wait_for_completion(client, data["job_id"])
    assert result["status"] == "failed"
    assert "disk full" in result["error"]


def test_invalid_container_upload_returns_actionable_422(monkeypatch):
    import subprocess
    monkeypatch.setattr(api.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "bad audio"))
    with TestClient(api.app) as client:
        result = client.post("/api/v1/analyze?background=true", headers=HEADERS, files={"audio_file": ("broken.mp4", b"broken", "video/mp4")})
        assert result.status_code == 422
        assert "decoding failed" in result.json()["detail"]


def test_legacy_job_keeps_ownership_between_analysis_and_conversion(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = api._validate_timeline
    monkeypatch.setattr(api, "separate_vocals", fake_separate_vocals)
    monkeypatch.setattr(api, "get_speech_cues", lambda _: [{"start_ms": 100.0625, "end_ms": 300.0625}])
    def hold_transition(record, submission):
        entered.set()
        assert release.wait(5)
        original(record, submission)
    monkeypatch.setattr(api, "_validate_timeline", hold_transition)
    with TestClient(api.app) as client:
        response = client.post("/api/v1/jobs", headers=HEADERS,
            files={"audio_file": ("test.wav", wav_bytes(), "audio/wav")},
            data={"job_data": json.dumps({"timeline": timeline("unused")["timeline"]})})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        try:
            assert entered.wait(2)
            with ThreadPoolExecutor(max_workers=2) as pool:
                conversion = pool.submit(client.post, "/api/v1/convert", headers=HEADERS, json=timeline(job_id))
                deletion = pool.submit(client.delete, f"/api/v1/jobs/{job_id}", headers=HEADERS)
                assert conversion.result().status_code == 409
                assert deletion.result().status_code == 409
            assert api.get_job(job_id).status == "processing"
        finally:
            release.set()
        assert wait_for_completion(client, job_id)["status"] == "completed"
        client.delete(f"/api/v1/jobs/{job_id}", headers=HEADERS)
