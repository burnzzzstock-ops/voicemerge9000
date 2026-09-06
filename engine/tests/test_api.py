from __future__ import annotations

import io
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

os.environ["VOICEMERGE_ENV"] = "test"
os.environ["VOICEMERGE_RVC_BACKEND"] = "copy"
os.environ["VOICEMERGE_JOB_ROOT"] = ".local/test-jobs"
os.environ["VOICEMERGE_MODEL_CACHE"] = ".local/test-models"
os.environ["VOICEMERGE_API_TOKEN"] = "test-token"

import engine.api as api_module  # noqa: E402


HEADERS = {"authorization": "Bearer test-token"}


def wav_bytes(sample_rate: int = 16_000, seconds: float = 0.5) -> bytes:
    samples = np.zeros(round(sample_rate * seconds), dtype=np.float32)
    start = round(sample_rate * 0.1)
    end = round(sample_rate * 0.3)
    samples[start:end] = np.sin(np.linspace(0, 80, end - start, dtype=np.float32)) * 0.15
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def fake_separate_vocals(audio_path: str, output_dir: str) -> tuple[str, str]:
    source, sample_rate = sf.read(audio_path, dtype="float32")
    destination = Path(output_dir)
    vocals = destination / "vocals.wav"
    background = destination / "background.wav"
    sf.write(vocals, source, sample_rate, subtype="FLOAT")
    sf.write(background, np.zeros((len(source), 2), dtype=np.float32), sample_rate, subtype="FLOAT")
    return str(vocals), str(background)


def wait_for_completion(client: TestClient, job_id: str) -> dict[str, object]:
    status: dict[str, object] | None = None
    for _ in range(100):
        status = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    assert status is not None
    return status


def test_analyze_convert_and_master_round_trip(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "separate_vocals", fake_separate_vocals)
    monkeypatch.setattr(
        api_module,
        "get_speech_cues",
        lambda _: [{"start_ms": 100.0, "end_ms": 300.0}],
    )
    converted_inputs: list[np.ndarray] = []
    original_convert = api_module.engine.convert_batch

    def capture_convert(sources, output_dir, model, params):
        converted_inputs.extend(sf.read(source, dtype="float32")[0] for source in sources)
        return original_convert(sources, output_dir, model, params)

    monkeypatch.setattr(api_module.engine, "convert_batch", capture_convert)
    submission = {
        "timeline": [{
            "speaker_id": "speaker-1",
            "model_id": "https://huggingface.co/example/model.pth",
            "gain": 1.0,
            "cues": [{"cue_id": "cue-00001", "start_ms": 100, "end_ms": 300}],
        }],
        "params": {"pitch": 0, "f0_method": "rmvpe", "index_rate": 0.75},
    }
    with TestClient(api_module.app) as client:
        assert client.get("/api/v1/health", headers=HEADERS).json()["backend"] == "copy"
        analysis = client.post(
            "/api/v1/analyze",
            headers=HEADERS,
            files={"audio_file": ("scene.wav", wav_bytes(), "audio/wav")},
        )
        assert analysis.status_code == 200
        analyzed = analysis.json()
        assert analyzed["cues"] == [{"cue_id": "cue-00001", "start_ms": 100, "end_ms": 300}]
        assert analyzed["speech_coverage"] == 0.4

        response = client.post(
            "/api/v1/convert",
            headers=HEADERS,
            json={"job_id": analyzed["job_id"], **submission},
        )
        assert response.status_code == 202
        status = wait_for_completion(client, analyzed["job_id"])
        assert status["status"] == "completed"
        assert status["progress"] == 1
        assert set(status["tracks"]) == {"vocals", "background", "master", "speaker-1"}
        assert len(converted_inputs) == 1
        assert len(converted_inputs[0]) == 9_600

        for track_name in ("speaker-1", "background", "master"):
            track = client.get(status["tracks"][track_name], headers=HEADERS)
            assert track.status_code == 200
            info = sf.info(io.BytesIO(track.content))
            assert info.frames == 24_000
            assert info.samplerate == 48_000
        master, _ = sf.read(io.BytesIO(client.get(status["tracks"]["master"], headers=HEADERS).content))
        assert np.max(np.abs(master)) <= 10 ** (-1.0 / 20.0) + 1e-6
        assert client.delete(f"/api/v1/jobs/{analyzed['job_id']}", headers=HEADERS).status_code == 204


def test_conversion_rejects_non_speech_timeline(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "separate_vocals", fake_separate_vocals)
    monkeypatch.setattr(
        api_module,
        "get_speech_cues",
        lambda _: [{"start_ms": 100.0, "end_ms": 300.0}],
    )
    with TestClient(api_module.app) as client:
        analysis = client.post(
            "/api/v1/analyze",
            headers=HEADERS,
            files={"audio_file": ("scene.wav", wav_bytes(), "audio/wav")},
        ).json()
        response = client.post(
            "/api/v1/convert",
            headers=HEADERS,
            json={
                "job_id": analysis["job_id"],
                "timeline": [{
                    "speaker_id": "speaker-1",
                    "model_id": "development",
                    "cues": [{"cue_id": "outside", "start_ms": 300, "end_ms": 450}],
                }],
            },
        )
        assert response.status_code == 422
        shutil.rmtree(api_module.get_job(analysis["job_id"]).directory, ignore_errors=True)
        with api_module.jobs_lock:
            api_module.jobs.pop(analysis["job_id"], None)
