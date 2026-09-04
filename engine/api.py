from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from . import __version__
from .audio import assemble_speaker_stem, inspect_audio, read_audio_mono, write_pcm24_wav
from .model_store import ModelArtifact, ModelStore
from .rvc_engine import RVCEngine
from .schemas import HealthResponse, JobCreated, JobProgress, JobSubmission


MAX_AUDIO_BYTES = int(os.getenv("VOICEMERGE_MAX_AUDIO", str(500 * 1024**2)))
MAX_DURATION_SECONDS = int(os.getenv("VOICEMERGE_MAX_DURATION_SECONDS", "7200"))
JOB_TTL_SECONDS = int(os.getenv("VOICEMERGE_JOB_TTL_SECONDS", "21600"))
JOB_ROOT = Path(os.getenv("VOICEMERGE_JOB_ROOT", "/tmp/voicemerge/jobs"))
API_TOKEN = os.getenv("VOICEMERGE_API_TOKEN")


@dataclass
class JobRecord:
    job_id: str
    directory: Path
    submission: JobSubmission
    status: str = "queued"
    progress: float = 0
    tracks: dict[str, str] = field(default_factory=dict)
    speaker_errors: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def response(self) -> JobProgress:
        with self.lock:
            return JobProgress(
                job_id=self.job_id,
                status=self.status,  # type: ignore[arg-type]
                progress=self.progress,
                tracks=dict(self.tracks),
                speaker_errors=dict(self.speaker_errors),
                error=self.error,
            )


app = FastAPI(title="VoiceMerge9000 RVC Engine", version=__version__)
allowed_origins = [value.strip() for value in os.getenv("VOICEMERGE_ALLOWED_ORIGINS", "http://localhost:3000").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

JOB_ROOT.mkdir(parents=True, exist_ok=True)
jobs: dict[str, JobRecord] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=int(os.getenv("VOICEMERGE_JOB_WORKERS", "2")))
gpu_slot = threading.Semaphore(int(os.getenv("VOICEMERGE_RVC_CONCURRENCY", "1")))
engine = RVCEngine()
model_store = ModelStore()


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid API token")


def purge_expired_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with jobs_lock:
        expired = [job_id for job_id, record in jobs.items() if record.created_at < cutoff and record.status in {"completed", "failed"}]
        for job_id in expired:
            record = jobs.pop(job_id)
            shutil.rmtree(record.directory, ignore_errors=True)


def get_job(job_id: str) -> JobRecord:
    purge_expired_jobs()
    with jobs_lock:
        record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="job not found or expired")
    return record


def development_model() -> ModelArtifact:
    return ModelArtifact(Path("development-copy.pth"), None, "development")


def process_job(record: JobRecord) -> None:
    with record.lock:
        record.status = "processing"
        record.progress = 0.01
    transient = record.directory / "work"
    tracks_directory = record.directory / "tracks"
    transient.mkdir(exist_ok=True)
    tracks_directory.mkdir(exist_ok=True)
    audio_path = record.directory / "source"
    completed_units = 0
    total_units = sum(len(speaker.cues) for speaker in record.submission.timeline)
    successful = 0
    try:
        metadata = inspect_audio(audio_path)
        master, sample_rate = read_audio_mono(audio_path)
        for speaker in record.submission.timeline:
            try:
                model = development_model() if engine.backend == "copy" else model_store.resolve(speaker.model_id)
                speaker_work = transient / speaker.speaker_id
                inputs = speaker_work / "inputs"
                outputs = speaker_work / "outputs"
                inputs.mkdir(parents=True)
                sources: list[Path] = []
                for cue in speaker.cues:
                    start = min(len(master), round(cue.start_ms * sample_rate / 1000))
                    end = min(len(master), round(cue.end_ms * sample_rate / 1000))
                    cue_input = inputs / f"{cue.cue_id}.wav"
                    write_pcm24_wav(cue_input, master[start:end], sample_rate)
                    sources.append(cue_input)
                with gpu_slot:
                    converted_files = engine.convert_batch(sources, outputs, model, record.submission.params)
                converted_cues: list[tuple[int, int, np.ndarray]] = []
                for cue, cue_input in zip(speaker.cues, sources, strict=True):
                    converted, _ = read_audio_mono(converted_files[cue_input], sample_rate)
                    converted_cues.append((cue.start_ms, cue.end_ms, converted))
                    completed_units += 1
                    with record.lock:
                        record.progress = min(0.96, completed_units / max(1, total_units) * 0.94 + 0.02)
                stem = assemble_speaker_stem(metadata.frames, sample_rate, converted_cues)
                track_path = tracks_directory / f"{speaker.speaker_id}.wav"
                write_pcm24_wav(track_path, stem, sample_rate)
                with record.lock:
                    record.tracks[speaker.speaker_id] = f"/api/v1/jobs/{record.job_id}/tracks/{speaker.speaker_id}.wav"
                successful += 1
            except Exception as error:
                completed_units += len(speaker.cues)
                with record.lock:
                    record.speaker_errors[speaker.speaker_id] = str(error)[:1000]
                    record.progress = min(0.96, completed_units / max(1, total_units) * 0.94 + 0.02)
        with record.lock:
            if successful:
                record.status = "completed"
                record.progress = 1
            else:
                record.status = "failed"
                record.progress = 1
                record.error = "No speaker tracks could be converted."
    except Exception as error:
        with record.lock:
            record.status = "failed"
            record.progress = 1
            record.error = str(error)[:1000]
    finally:
        shutil.rmtree(transient, ignore_errors=True)


@app.get("/api/v1/health", response_model=HealthResponse)
def health(_: None = Depends(require_token)) -> HealthResponse:
    engine.ready()
    return HealthResponse(version=__version__, backend=engine.backend, device=engine.device())


@app.post("/api/v1/jobs", response_model=JobCreated, status_code=202)
async def create_job(
    audio_file: Annotated[UploadFile, File()],
    job_data: Annotated[str, Form()],
    _: None = Depends(require_token),
) -> JobCreated:
    purge_expired_jobs()
    try:
        submission = JobSubmission.model_validate_json(job_data)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=json.loads(error.json())) from error
    job_id = uuid.uuid4().hex
    directory = JOB_ROOT / job_id
    directory.mkdir(parents=True)
    audio_path = directory / "source"
    size = 0
    try:
        with audio_path.open("wb") as target:
            while chunk := await audio_file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="audio file exceeds the upload limit")
                target.write(chunk)
        metadata = inspect_audio(audio_path)
        if metadata.duration_seconds > MAX_DURATION_SECONDS:
            raise HTTPException(status_code=413, detail="audio duration exceeds the job limit")
        duration_ms = round(metadata.duration_seconds * 1000)
        for speaker in submission.timeline:
            if speaker.cues[-1].end_ms > duration_ms + 2:
                raise HTTPException(status_code=422, detail=f"{speaker.speaker_id} contains a cue beyond the audio duration")
        record = JobRecord(job_id=job_id, directory=directory, submission=submission)
        with jobs_lock:
            jobs[job_id] = record
        executor.submit(process_job, record)
        return JobCreated(job_id=job_id, status="queued")
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await audio_file.close()


@app.get("/api/v1/jobs/{job_id}", response_model=JobProgress)
def read_job(job_id: str, _: None = Depends(require_token)) -> JobProgress:
    return get_job(job_id).response()


@app.get("/api/v1/jobs/{job_id}/tracks/{speaker_id}.wav")
def read_track(job_id: str, speaker_id: str, _: None = Depends(require_token)) -> FileResponse:
    record = get_job(job_id)
    if speaker_id not in record.tracks:
        raise HTTPException(status_code=404, detail="speaker track is not ready")
    path = record.directory / "tracks" / f"{speaker_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="speaker track has expired")
    return FileResponse(path, media_type="audio/wav", filename=f"{speaker_id}.wav")


@app.delete("/api/v1/jobs/{job_id}", status_code=204, response_class=Response)
def delete_job(job_id: str, _: None = Depends(require_token)) -> Response:
    record = get_job(job_id)
    if record.status in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="a running job cannot be deleted")
    with jobs_lock:
        jobs.pop(job_id, None)
    shutil.rmtree(record.directory, ignore_errors=True)
    return Response(status_code=204)
