from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

os.environ["VOICEMERGE_ENV"] = "test"
os.environ["VOICEMERGE_RVC_BACKEND"] = "copy"
os.environ["VOICEMERGE_JOB_ROOT"] = ".local/test-jobs"
os.environ["VOICEMERGE_MODEL_CACHE"] = ".local/test-models"
os.environ["VOICEMERGE_API_TOKEN"] = "test-token"

from engine.api import app  # noqa: E402


HEADERS = {"authorization": "Bearer test-token"}


def wav_bytes(sample_rate: int = 16_000, seconds: float = 0.5) -> bytes:
    samples = np.sin(np.linspace(0, 80, round(sample_rate * seconds), dtype=np.float32)) * 0.15
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def test_copy_backend_job_round_trip() -> None:
    submission = {
        "timeline": [{
            "speaker_id": "speaker-1",
            "model_id": "https://huggingface.co/example/model.pth",
            "cues": [{"cue_id": "cue-1", "start_ms": 100, "end_ms": 300}],
        }],
        "params": {"pitch": 0, "f0_method": "rmvpe", "index_rate": 0.75},
    }
    with TestClient(app) as client:
        assert client.get("/api/v1/health", headers=HEADERS).json()["backend"] == "copy"
        response = client.post(
            "/api/v1/jobs",
            headers=HEADERS,
            files={"audio_file": ("scene.wav", wav_bytes(), "audio/wav")},
            data={"job_data": json.dumps(submission)},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        status = None
        for _ in range(100):
            status = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()
            if status["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert status is not None
        assert status["status"] == "completed"
        assert status["progress"] == 1
        track = client.get(status["tracks"]["speaker-1"], headers=HEADERS)
        assert track.status_code == 200
        info = sf.info(io.BytesIO(track.content))
        assert info.frames == 8_000
        assert info.samplerate == 16_000
        assert client.delete(f"/api/v1/jobs/{job_id}", headers=HEADERS).status_code == 204
