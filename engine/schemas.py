from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CueSegment(StrictModel):
    cue_id: str = Field(pattern=ID_PATTERN)
    start_ms: int = Field(ge=0, le=14_400_000)
    end_ms: int = Field(gt=0, le=14_400_000)

    @model_validator(mode="after")
    def validate_range(self) -> "CueSegment":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.end_ms - self.start_ms < 40:
            raise ValueError("a cue must be at least 40 ms long")
        return self


class SpeakerTimeline(StrictModel):
    speaker_id: str = Field(pattern=ID_PATTERN)
    model_id: str = Field(min_length=1, max_length=4096)
    gain: float = Field(default=1.0, ge=0.25, le=1.75)
    cues: list[CueSegment] = Field(min_length=1, max_length=10_000)

    @field_validator("speaker_id")
    @classmethod
    def reject_reserved_track_names(cls, speaker_id: str) -> str:
        if speaker_id.lower() in {"background", "master", "vocals"}:
            raise ValueError("speaker_id is reserved for an engine track")
        return speaker_id

    @field_validator("cues")
    @classmethod
    def validate_cues(cls, cues: list[CueSegment]) -> list[CueSegment]:
        ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms))
        if len({cue.cue_id for cue in ordered}) != len(ordered):
            raise ValueError("cue_id values must be unique within a speaker")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.start_ms < previous.end_ms:
                raise ValueError("cues for one speaker may not overlap")
        return ordered


class RVCParams(StrictModel):
    pitch: int = Field(default=0, ge=-24, le=24)
    f0_method: Literal["rmvpe", "pm"] = "rmvpe"
    index_rate: float = Field(default=0.75, ge=0, le=1)
    filter_radius: int = Field(default=3, ge=0, le=7)
    rms_mix_rate: float = Field(default=0.25, ge=0, le=1)
    protect: float = Field(default=0.33, ge=0, le=0.5)


class JobSubmission(StrictModel):
    timeline: list[SpeakerTimeline] = Field(min_length=1, max_length=6)
    params: RVCParams = Field(default_factory=RVCParams)

    @field_validator("timeline")
    @classmethod
    def validate_speakers(cls, timeline: list[SpeakerTimeline]) -> list[SpeakerTimeline]:
        ids = [speaker.speaker_id for speaker in timeline]
        if len(set(ids)) != len(ids):
            raise ValueError("speaker_id values must be unique")
        return timeline


class ConversionRequest(JobSubmission):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class SpeechCueResult(StrictModel):
    cue_id: str = Field(pattern=ID_PATTERN)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)


JobStatus = Literal["queued", "processing", "analyzed", "completed", "failed"]


class JobCreated(StrictModel):
    job_id: str
    status: JobStatus


class JobProgress(StrictModel):
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0, le=1)
    tracks: dict[str, str] = Field(default_factory=dict)
    speaker_errors: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    stage: str | None = None
    active_speaker: str | None = None
    downloaded_bytes: int | None = Field(default=None, ge=0)
    total_bytes: int | None = Field(default=None, ge=0)


class AnalysisResponse(StrictModel):
    job_id: str
    status: Literal["analyzed"] = "analyzed"
    duration_ms: int = Field(gt=0)
    sample_rate: int = Field(gt=0)
    speech_coverage: float = Field(ge=0, le=1)
    cues: list[SpeechCueResult]


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    version: str
    backend: str
    device: str
