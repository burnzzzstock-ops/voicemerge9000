from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from . import __version__
from .audio import (
    assemble_speaker_stem,
    extract_cue_from_array,
    inspect_audio,
    mix_background_and_stems,
    pad_or_trim_clip,
    peak_limit_and_normalize,
    read_audio,
    read_audio_mono,
    write_float_wav,
    write_pcm24_wav,
)
from .model_store import ModelArtifact, ModelStore
from .rvc_engine import RVCEngine
from .schemas import (
    AnalysisResponse,
    ConversionRequest,
    HealthResponse,
    JobCreated,
    JobProgress,
    JobSubmission,
    SpeechCueResult,
)
from .separator import separate_vocals
from .vad import get_speech_cues


MAX_AUDIO_BYTES = int(os.getenv("VOICEMERGE_MAX_AUDIO", str(500 * 1024**2)))
MAX_DURATION_SECONDS = int(os.getenv("VOICEMERGE_MAX_DURATION_SECONDS", "7200"))
JOB_TTL_SECONDS = int(os.getenv("VOICEMERGE_JOB_TTL_SECONDS", "21600"))
JOB_ROOT = Path(os.getenv("VOICEMERGE_JOB_ROOT", "/tmp/voicemerge/jobs"))
API_TOKEN = os.getenv("VOICEMERGE_API_TOKEN")
TARGET_SAMPLE_RATE = int(os.getenv("VOICEMERGE_SAMPLE_RATE", "48000"))


@dataclass
class JobRecord:
    job_id: str
    directory: Path
    duration_ms: int
    sample_rate: int
    total_samples: int
    channels: int
    submission: JobSubmission | None = None
    detected_cues: list[SpeechCueResult] = field(default_factory=list)
    status: str = "queued"
    progress: float = 0
    tracks: dict[str, str] = field(default_factory=dict)
    speaker_errors: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    stage: str | None = "queued"
    active_speaker: str | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    created_at: float = field(default_factory=time.time)
    attempt_id: str | None = None
    output_files: dict[tuple[str, str], Path] = field(default_factory=dict)
    speaker_cache: dict[str, Path] = field(default_factory=dict)
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
                stage=self.stage,
                active_speaker=self.active_speaker,
                downloaded_bytes=self.downloaded_bytes,
                total_bytes=self.total_bytes,
                attempt_id=self.attempt_id,
                analysis=self.analysis_response() if self.status == "analyzed" else None,
            )

    def analysis_response(self) -> AnalysisResponse:
        speech_ms = sum(cue.end_ms - cue.start_ms for cue in self.detected_cues)
        return AnalysisResponse(
            job_id=self.job_id,
            duration_ms=self.duration_ms,
            sample_rate=self.sample_rate,
            speech_coverage=min(1.0, speech_ms / max(1, self.duration_ms)),
            cues=list(self.detected_cues),
            tracks={key: _track_url(self.job_id, key) for key in ("vocals", "background")},
        )


app = FastAPI(title="VoiceMerge9000 RVC Engine", version=__version__)
allowed_origins = [
    value.strip()
    for value in os.getenv("VOICEMERGE_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if value.strip()
]
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
gpu_slot = threading.Semaphore(1)
engine = RVCEngine()
model_store = ModelStore()


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid API token")


def purge_expired_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    terminal_states = {"analyzed", "completed", "failed"}
    with jobs_lock:
        for job_id, record in list(jobs.items()):
            with record.lock:
                if record.created_at < cutoff and record.status in terminal_states:
                    jobs.pop(job_id)
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


def _track_url(job_id: str, track_name: str, attempt_id: str | None = None) -> str:
    suffix = f"?attempt_id={attempt_id}" if attempt_id else ""
    return f"/api/v1/jobs/{job_id}/tracks/{track_name}.wav{suffix}"


def _fail_job(record: JobRecord, error: Exception | str) -> None:
    with record.lock:
        record.status = "failed"
        record.progress = 1
        record.error = str(error)[:1000]
        record.stage = "failed"
        record.active_speaker = None
        record.downloaded_bytes = None
        record.total_bytes = None


def _normalize_separated_stems(record: JobRecord, vocals_path: str, background_path: str) -> None:
    vocals, _ = read_audio(vocals_path, record.sample_rate)
    vocals_mono = np.mean(vocals, axis=1, dtype=np.float32)
    vocals_mono = pad_or_trim_clip(vocals_mono, record.total_samples, record.sample_rate)

    background, _ = read_audio(background_path, record.sample_rate)
    if record.channels == 1:
        background = np.mean(background, axis=1, dtype=np.float32)[:, None]
    elif background.shape[1] == 1:
        background = np.repeat(background, 2, axis=1)
    else:
        background = background[:, :2]
    background = pad_or_trim_clip(background, record.total_samples, record.sample_rate)

    write_float_wav(record.directory / "vocals.wav", vocals_mono, record.sample_rate)
    write_float_wav(record.directory / "background.wav", background, record.sample_rate)


def _analyze_record(record: JobRecord, *, for_conversion: bool = False) -> None:
    try:
        with record.lock:
            record.status = "processing"
            record.progress = 0.05
            record.stage = "queued_separation"
            record.error = None
        with gpu_slot:
            with record.lock:
                record.stage = "separating_vocals"
            vocals_path, background_path = separate_vocals(
                str(record.directory / ("source.wav" if (record.directory / "source.wav").is_file() else "source")),
                str(record.directory),
            )
        _normalize_separated_stems(record, vocals_path, background_path)

        with record.lock:
            record.progress = 0.85
            record.stage = "detecting_speech"
        detected = get_speech_cues(str(record.directory / "vocals.wav"))
        cues: list[SpeechCueResult] = []
        for item in detected:
            exact_duration_ms = record.total_samples * 1000.0 / record.sample_rate
            start_ms = max(0, min(exact_duration_ms, item["start_ms"]))
            end_ms = max(start_ms, min(exact_duration_ms, item["end_ms"]))
            if end_ms - start_ms >= 40:
                cues.append(
                    SpeechCueResult(
                        cue_id=f"cue-{len(cues) + 1:05d}",
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
            if len(cues) >= 10_000:
                break
        with record.lock:
            record.detected_cues = cues
            # Legacy upload+convert owns the job across this transition.
            # Publishing a terminal state here would permit a competing retry.
            record.status = "processing" if for_conversion else "analyzed"
            record.progress = 0.9 if for_conversion else 1
            record.stage = "preparing_vocals" if for_conversion else "analyzed"
            record.tracks = {key: _track_url(record.job_id, key) for key in ("vocals", "background")}
    except Exception as error:
        _fail_job(record, error)
        raise


def _validate_timeline(record: JobRecord, submission: JobSubmission) -> None:
    all_cues = [cue for speaker in submission.timeline for cue in speaker.cues]
    ordered = sorted(all_cues, key=lambda cue: (cue.start_ms, cue.end_ms))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.start_ms < previous.end_ms:
            raise ValueError("speaker cues may not overlap across the cast")
    for cue in ordered:
        # Tolerate JSON/browser seconds-to-ms round-off, not an extra audio sample.
        if cue.end_ms > record.total_samples * 1000.0 / record.sample_rate + 1e-7:
            raise ValueError(f"{cue.cue_id} extends beyond the scene duration")
        is_confirmed_speech = any(
            cue.start_ms >= detected.start_ms - 150 - 1e-7 and cue.end_ms <= detected.end_ms + 150 + 1e-7
            and cue.start_ms < detected.end_ms and cue.end_ms > detected.start_ms
            for detected in record.detected_cues
        )
        if not is_confirmed_speech:
            raise ValueError(f"{cue.cue_id} is outside the isolated speech regions")


def process_job(record: JobRecord) -> None:
    submission = record.submission
    if submission is None:
        _fail_job(record, "conversion has no timeline")
        return
    with record.lock:
        record.status = "processing"
        record.progress = 0.01
        record.stage = "preparing_vocals"
    attempt_id = record.attempt_id or uuid.uuid4().hex
    record.attempt_id = attempt_id
    attempt_directory = record.directory / "attempts" / attempt_id
    transient = attempt_directory / "work"
    tracks_directory = attempt_directory / "tracks"
    completed_units = 0
    total_units = sum(len(speaker.cues) for speaker in submission.timeline)
    successful_stems: list[np.ndarray] = []
    try:
        transient.mkdir(parents=True, exist_ok=True)
        tracks_directory.mkdir(parents=True, exist_ok=True)
        vocals, sample_rate = read_audio_mono(record.directory / "vocals.wav", record.sample_rate)
        vocals = pad_or_trim_clip(vocals, record.total_samples, sample_rate)
        for speaker in submission.timeline:
            speaker_units = len(speaker.cues)
            cache_key = hashlib.sha256(json.dumps({
                "speaker": speaker.model_dump(exclude={"gain"}), "params": submission.params.model_dump(),
                "sample_rate": sample_rate, "total_samples": record.total_samples,
            }, sort_keys=True).encode()).hexdigest()
            cached = record.speaker_cache.get(cache_key)
            if cached and cached.is_file():
                stem, _ = read_audio_mono(cached, sample_rate)
                successful_stems.append(stem * np.float32(speaker.gain))
                completed_units += speaker_units
                with record.lock:
                    record.output_files[(attempt_id, speaker.speaker_id)] = cached
                    record.tracks[speaker.speaker_id] = _track_url(record.job_id, speaker.speaker_id, attempt_id)
                continue
            base_progress = min(0.88, completed_units / max(1, total_units) * 0.86 + 0.02)
            speaker_span = speaker_units / max(1, total_units) * 0.86

            def report_model_progress(stage: str, downloaded: int | None, total: int | None) -> None:
                with record.lock:
                    record.stage = stage
                    record.active_speaker = speaker.speaker_id
                    record.downloaded_bytes = downloaded
                    record.total_bytes = total
                    if stage == "downloading_model" and downloaded is not None and total:
                        fraction = min(1, downloaded / total)
                        record.progress = max(
                            record.progress,
                            min(0.88, base_progress + speaker_span * 0.2 * fraction),
                        )

            try:
                model = (
                    development_model()
                    if engine.backend == "copy"
                    else model_store.resolve(speaker.model_id, report_model_progress)
                )
                speaker_work = transient / speaker.speaker_id
                inputs = speaker_work / "inputs"
                outputs = speaker_work / "outputs"
                inputs.mkdir(parents=True)
                sources: list[Path] = []
                for cue in speaker.cues:
                    cue_input = inputs / f"{cue.cue_id}.wav"
                    dry_cue = extract_cue_from_array(vocals, cue.start_ms, cue.end_ms, sample_rate)
                    write_pcm24_wav(cue_input, dry_cue, sample_rate)
                    sources.append(cue_input)
                with record.lock:
                    record.stage = "converting_speech"
                    record.active_speaker = speaker.speaker_id
                    record.downloaded_bytes = None
                    record.total_bytes = None
                    record.progress = max(record.progress, min(0.88, base_progress + speaker_span * 0.2))
                with gpu_slot:
                    try:
                        converted_files = engine.convert_batch(sources, outputs, model, submission.params)
                    finally:
                        if (outputs / ".gpu.json").is_file():
                            shutil.copyfile(outputs / ".gpu.json", attempt_directory / f"gpu-{speaker.speaker_id}.json")
                converted_cues: list[tuple[float, float, np.ndarray]] = []
                for cue, cue_input in zip(speaker.cues, sources, strict=True):
                    converted, _ = read_audio_mono(converted_files[cue_input], sample_rate)
                    converted_cues.append((cue.start_ms, cue.end_ms, converted))
                    completed_units += 1
                    with record.lock:
                        record.progress = min(
                            0.88,
                            completed_units / max(1, total_units) * 0.86 + 0.02,
                        )
                with record.lock:
                    record.stage = "assembling_speaker_stem"
                stem = assemble_speaker_stem(converted_cues, record.total_samples, sample_rate)
                # Cache canonical pre-fader audio. Per-voice peak protection must
                # not cancel a later mix-gain adjustment or force new inference.
                stem = peak_limit_and_normalize(
                    stem,
                    target_lufs=-18.0,
                    peak_limit=-1.0,
                    sample_rate=sample_rate,
                )
                track_path = tracks_directory / f"{speaker.speaker_id}.wav"
                write_pcm24_wav(track_path, stem, sample_rate)
                # Use the same quantized samples on fresh and cached paths.
                canonical_stem, _ = read_audio_mono(track_path, sample_rate)
                successful_stems.append(canonical_stem * np.float32(speaker.gain))
                with record.lock:
                    record.output_files[(attempt_id, speaker.speaker_id)] = track_path
                    record.speaker_cache[cache_key] = track_path
                    record.tracks[speaker.speaker_id] = _track_url(record.job_id, speaker.speaker_id, attempt_id)
            except Exception as error:
                completed_units += speaker_units
                with record.lock:
                    record.speaker_errors[speaker.speaker_id] = str(error)[:1000]
                    record.progress = min(
                        0.88,
                        completed_units / max(1, total_units) * 0.86 + 0.02,
                    )

        if record.speaker_errors:
            raise RuntimeError("Some voices failed. No final master was created because dialogue would be missing. Retry to reuse successful voices.")
        if not successful_stems:
            raise RuntimeError("No speaker tracks could be converted.")

        with record.lock:
            record.progress = 0.9
            record.stage = "mixing_master"
            record.active_speaker = None
        background, _ = read_audio(record.directory / "background.wav", sample_rate)
        background = pad_or_trim_clip(background, record.total_samples, sample_rate)
        master = mix_background_and_stems(
            background,
            successful_stems,
            sample_rate,
            target_lufs=-16.0,
            peak_limit=-1.0,
        )
        master_path = tracks_directory / "master.wav"
        write_pcm24_wav(master_path, master, sample_rate)
        with record.lock:
            record.tracks["background"] = _track_url(record.job_id, "background")
            record.output_files[(attempt_id, "master")] = master_path
            record.tracks["master"] = _track_url(record.job_id, "master", attempt_id)
            record.status = "completed"
            record.progress = 1
            record.stage = "completed"
            record.active_speaker = None
            record.downloaded_bytes = None
            record.total_bytes = None
    except Exception as error:
        _fail_job(record, error)
    finally:
        shutil.rmtree(transient, ignore_errors=True)


def _process_legacy_job(record: JobRecord) -> None:
    try:
        _analyze_record(record, for_conversion=True)
        if record.submission is None:
            raise ValueError("conversion has no timeline")
        _validate_timeline(record, record.submission)
        process_job(record)
    except Exception as error:
        _fail_job(record, error)


async def _store_upload(audio_file: UploadFile) -> JobRecord:
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
        try:
            metadata = inspect_audio(audio_path)
        except (RuntimeError, ValueError):
            canonical = directory / "source.wav"
            try:
                decoded = await run_in_threadpool(subprocess.run, [
                    "ffmpeg", "-nostdin", "-v", "error", "-i", str(audio_path),
                    "-map", "0:a:0", "-vn", "-t", str(MAX_DURATION_SECONDS + 1),
                    "-ar", str(TARGET_SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_f32le", str(canonical),
                ], capture_output=True, text=True, timeout=300, check=False)
                if decoded.returncode:
                    raise ValueError("The file has no supported audio stream or is damaged.")
                metadata = inspect_audio(canonical)
            except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
                raise HTTPException(status_code=422, detail="Audio decoding failed: no supported audio stream, damaged file, or FFmpeg unavailable. Try an MP3/WAV.") from error
        if metadata.duration_seconds > MAX_DURATION_SECONDS:
            raise HTTPException(status_code=413, detail="audio duration exceeds the job limit")
        return JobRecord(
            job_id=job_id,
            directory=directory,
            duration_ms=round(metadata.duration_seconds * 1000),
            sample_rate=TARGET_SAMPLE_RATE,
            total_samples=round(metadata.duration_seconds * TARGET_SAMPLE_RATE),
            channels=metadata.channels,
        )
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await audio_file.close()


@app.get("/api/v1/health", response_model=HealthResponse)
def health(_: None = Depends(require_token)) -> HealthResponse:
    engine.ready()
    return HealthResponse(version=__version__, backend=engine.backend, device=engine.device())


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_scene(
    audio_file: Annotated[UploadFile, File()],
    background: bool = False,
    _: None = Depends(require_token),
) -> AnalysisResponse | Response:
    purge_expired_jobs()
    record = await _store_upload(audio_file)
    with jobs_lock:
        jobs[record.job_id] = record
    if background:
        try:
            executor.submit(_analyze_record, record)
        except RuntimeError as error:
            _fail_job(record, "The analysis worker could not start. Please upload again.")
            raise HTTPException(status_code=503, detail=record.error) from error
        return Response(JobCreated(job_id=record.job_id, status="queued").model_dump_json(), status_code=202, media_type="application/json")
    try:
        await run_in_threadpool(_analyze_record, record)
        return record.analysis_response()
    except Exception as error:
        with jobs_lock:
            jobs.pop(record.job_id, None)
        shutil.rmtree(record.directory, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"scene analysis failed: {str(error)[:500]}") from error


@app.post("/api/v1/convert", response_model=JobCreated, status_code=202)
def convert_scene(
    request: ConversionRequest,
    _: None = Depends(require_token),
) -> JobCreated:
    record = get_job(request.job_id)
    with jobs_lock, record.lock:
        if jobs.get(record.job_id) is not record:
            raise HTTPException(status_code=404, detail="job expired; upload again")
        if record.status in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="job is already running")
        if not (record.directory / "vocals.wav").is_file():
            raise HTTPException(status_code=409, detail="Analyze the scene again before converting.")
        submission = JobSubmission(timeline=request.timeline, params=request.params).model_copy(deep=True)
        try:
            _validate_timeline(record, submission)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        record.submission = submission
        record.attempt_id = uuid.uuid4().hex
        record.status = "queued"
        record.progress = 0
        record.stage = "queued"
        record.tracks = {key: _track_url(record.job_id, key) for key in ("vocals", "background")}
        record.speaker_errors.clear()
        record.error = None
        record.created_at = time.time()
    try:
        executor.submit(process_job, record)
    except RuntimeError as error:
        _fail_job(record, "The worker could not start. Please retry.")
        raise HTTPException(status_code=503, detail=record.error) from error
    return JobCreated(job_id=record.job_id, status="queued", attempt_id=record.attempt_id)


@app.post("/api/v1/jobs", response_model=JobCreated, status_code=202)
async def create_legacy_job(
    audio_file: Annotated[UploadFile, File()],
    job_data: Annotated[str, Form()],
    _: None = Depends(require_token),
) -> JobCreated:
    """Compatibility endpoint; new clients should call analyze, then convert."""
    purge_expired_jobs()
    try:
        submission = JobSubmission.model_validate_json(job_data)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=json.loads(error.json())) from error
    record = await _store_upload(audio_file)
    record.submission = submission
    with jobs_lock:
        jobs[record.job_id] = record
    try:
        executor.submit(_process_legacy_job, record)
    except RuntimeError as error:
        _fail_job(record, "The worker could not start. Please upload again.")
        raise HTTPException(status_code=503, detail=record.error) from error
    return JobCreated(job_id=record.job_id, status="queued")


@app.get("/api/v1/jobs/{job_id}", response_model=JobProgress)
def read_job(job_id: str, _: None = Depends(require_token)) -> JobProgress:
    return get_job(job_id).response()


@app.get("/api/v1/jobs/{job_id}/tracks/{track_name}")
def read_track(job_id: str, track_name: str, attempt_id: str | None = None, _: None = Depends(require_token)) -> FileResponse:
    record = get_job(job_id)
    key = track_name[:-4] if track_name.lower().endswith(".wav") else track_name
    with record.lock:
        path = (record.directory / f"{key}.wav" if key in {"background", "vocals"}
                else record.output_files.get((attempt_id or record.attempt_id or "", key)))
    if path is None:
        raise HTTPException(status_code=404, detail="track is not ready")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="track has expired")
    return FileResponse(path, media_type="audio/wav", filename=f"{key}.wav")


@app.delete("/api/v1/jobs/{job_id}", status_code=204, response_class=Response)
def delete_job(job_id: str, _: None = Depends(require_token)) -> Response:
    record = get_job(job_id)
    with jobs_lock, record.lock:
        if record.status in {"queued", "processing"}:
            raise HTTPException(status_code=409, detail="a running job cannot be deleted")
        jobs.pop(job_id, None)
        shutil.rmtree(record.directory, ignore_errors=True)
    return Response(status_code=204)
