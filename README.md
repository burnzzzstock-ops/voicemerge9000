# VoiceMerge9000

> **Experimental, unfinished — contributors and maintainers welcome.** The local pipeline produces real voice conversions, but can miss vocals, sound robotic, and mix voices unevenly. This is a project handoff, not a production-quality release. Start with [the maintainer handoff and prioritized roadmap](HANDOFF.md).

VoiceMerge9000 is a one-screen character voice casting desk for scenes, songs, and dialogue. Upload one audio file, review who speaks where, assign a different RVC model to each speaker, then click **Convert all + merge**. The local engine separates the soundtrack, converts selected detected vocal sections, rebuilds full-length speaker stems at their original timestamps, and mixes them with the separated stereo background. Detection and speaker assignments are guesses requiring review; singing and entirely missed sections remain important limitations.

Use fictional, stylized, public-domain, or otherwise authorized voices and media. The source repository does not bundle character voice models and must not be used to impersonate a real person without permission. Docker builds download pretrained inference weights: the project's MIT license does not grant rights to those weights, community models, or input media. Audit their terms separately before redistributing images or assets.

## What is implemented

- `htdemucs` two-stem vocal/background separation with local GPU or CPU execution
- Silero VAD over the isolated vocal stem and an editable acoustic speaker draft
- One to six independent Voice A–F character assignments
- In-app `voice-models.com` search, samples, metadata, and direct model selection
- A same-origin job API: the user never has to open a second converter
- A typed two-stage FastAPI service with analysis, conversion, progress, background/master/per-speaker tracks, and deletion endpoints
- Official RVC CLI inference, CUDA detection with CPU fallback, and one model load per speaker batch
- Sample-accurate cue extraction, zero-padding/boundary trimming, 5 ms micro-fades, silent full-length stems, and an engine-rendered 48 kHz / 24-bit master
- LUFS normalization and oversampled −1 dBFS true-peak protection without waveform interpolation
- Failure isolation: one failed speaker does not discard successful tracks
- Model cache and trust boundary: HTTPS host allowlisting with optional HMAC-signed IDs/checksums, redirect/DNS checks, size ceilings, safe archive extraction, SHA-256 object storage, and PyTorch weights-only preflight
- A development copy backend for timing/API tests; separation still requires Demucs/Silero and their model weights

Silero decides where speech exists; the Voice A–F assignment is still a lightweight acoustic guess, not identity recognition or full diarization. Crosstalk and similar voices can confuse it, so human review is intentionally part of the workflow.

## Easiest local run

Requirements: Docker, Docker Compose, and enough disk space for the RVC runtime and model cache.

1. Copy `.env.example` to `.env` and replace both `change-me` values with the same long random token.
2. Start the app:

```bash
docker compose up --build
```

3. Open `http://localhost:3000`. Only the VoiceMerge interface is exposed; the Python worker stays on Docker's private network.

The first upload runs vocal separation before the review timeline appears. Demucs and Silero weights are cached in the image during the build, so this stage does not download models at runtime.

The default compose file runs on CPU when no compatible GPU is visible. NVIDIA users with the Container Toolkit installed can enable CUDA:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Fast development loop

Run the engine's safe copy backend (it preserves timing but does not change timbre):

```bash
python -m venv .venv
.venv/bin/pip install -r engine/requirements-dev.txt
VOICEMERGE_ENV=development VOICEMERGE_RVC_BACKEND=copy \
VOICEMERGE_JOB_ROOT=.local/jobs VOICEMERGE_MODEL_CACHE=.local/models \
VOICEMERGE_API_TOKEN=change-me .venv/bin/uvicorn engine.api:app --port 8000
```

In a second terminal:

```bash
pnpm install
RVC_ENGINE_URL=http://127.0.0.1:8000 RVC_ENGINE_API_TOKEN=change-me pnpm dev
```

Windows PowerShell uses `$env:NAME='value'` before each command and `.venv/Scripts/` instead of `.venv/bin/`. Install FFmpeg and make it available on PATH. Open `http://localhost:3000`.

The development requirements include CPU PyTorch, Demucs, and Silero. The first analysis downloads separator weights unless already cached. Copy mode does not change voices and is not audio-quality verification. Use the Docker GPU configuration for real RVC; it keeps the legacy RVC dependencies separate from the API environment.

Docker publishes the web app on loopback only (127.0.0.1), not to other devices on your network. Do not expose the proxy publicly without adding user authentication and appropriate access controls.

Tests and frontend checks:

```bash
.venv/bin/pytest engine/tests
pnpm test
pnpm exec tsc --noEmit
pnpm run lint
pnpm build
```

## Engine API

- `GET /api/v1/health`
- `POST /api/v1/analyze` with multipart `audio_file`; returns the reusable `job_id`, speech cues, duration, and speech coverage
- `POST /api/v1/analyze?background=true` returns HTTP 202 immediately; poll the job for stages and its `analysis` metadata, including isolated-vocal/background preview URLs
- `POST /api/v1/convert` with the `job_id`, reviewed speaker timeline, model IDs, and RVC parameters
- `GET /api/v1/jobs/{job_id}` for progress and track URLs
- `GET /api/v1/jobs/{job_id}/tracks/{track_name}` for `master.wav`, `background.wav`, or a speaker WAV
- `DELETE /api/v1/jobs/{job_id}` after the tracks are downloaded

The older multipart `POST /api/v1/jobs` route remains as a compatibility shim. It now runs separation and VAD before conversion too; mixed soundtrack audio is never sent directly to RVC.

Conversion attempts have immutable URLs containing `attempt_id`. A failed voice blocks the final master while retaining successful previews. Retrying the same cast reuses matching successful stems. Changing a model, timing, or RVC parameters invalidates the relevant cached voice. Gain-only edits reuse canonical normalized stems and apply mix gain during master assembly; individual voice previews/downloads are before mix gain. Final mastering applies a common gain, so the control changes the voice's relative balance rather than guaranteeing an absolute output loudness. Earlier downloads remain available until the job expires or is deleted. Job metadata is currently in memory: restarting the engine invalidates active job IDs, so download finished audio before restarting.

## Review and audio-quality limits

- Confirm the speaker guesses; VAD is not diarization. Preview original audio, isolated vocals, background, and individual converted speakers.
- Timing edits may extend up to 150 ms beyond detected speech, must overlap detected speech, and cannot overlap another speaker. Split longer sections to correct speaker changes.
- Separation may leave background bleed, mistake singing for dialogue, or remove vocal sounds outside selected regions. Overlapping speakers are not independently separated. Human review remains necessary.
- Outputs are 48 kHz / 24-bit WAV. Mastering uses bounded loudness gain and a 4× oversampled peak estimate with a −1 dBFS ceiling. High-crest-factor material may finish below the −16 LUFS master target; this is not a broadcast-certified dynamic limiter.
- Demucs and RVC run sequentially in short-lived workers. A 6.5 GiB budget reserves 0.5 GiB headroom and caps PyTorch allocations before model loading; CUDA graphs are disabled for RVC. This does not guarantee total device usage: other apps, driver/context allocations, and non-PyTorch allocations must be measured separately. Per-stage allocator metrics are stored in the local job directory.
- If Docker reports `WSL environment detected but no adapters were found`, GPU inference cannot start. Restore NVIDIA GPU visibility in Windows/WSL and restart Docker Desktop before retrying. Do not mistake a copy-backend test for successful voice conversion.

## Third-party licenses

This repository's MIT license applies to its own code, not every dependency or model. RVC, Demucs, Silero, PyTorch, FFmpeg, and the NVIDIA CUDA container retain their respective licenses. FFmpeg licensing depends on the installed build. Community voice models retain their creators' terms and may have no redistribution permission. Do not bundle checkpoints, indexes, source scenes, or generated test audio into source releases; verify the applicable model and media permissions separately.

The browser calls `/api/engine/*`; the web app proxies that to `RVC_ENGINE_URL` and injects the private engine token. Set `VOICEMERGE_MODEL_HOSTS` to the smallest host allowlist you need. Production model registries can additionally issue `v1.<payload>.<hmac>` model IDs containing an expiry, allowlisted URL, and optional SHA-256.

## Deployment

The web interface can run on Sites or any compatible JavaScript host. RVC requires Python, FFmpeg, large model files, and usually CUDA, so deploy `engine/Dockerfile` to a GPU/container host and set the web deployment's private `RVC_ENGINE_URL` and `RVC_ENGINE_API_TOKEN`. This still presents one app and one button to the user; the inference worker is an internal service.

## Privacy

- Source audio reaches the configured VoiceMerge engine when it is uploaded for local Demucs/Silero analysis.
- The engine keeps the isolated stems under an opaque job ID and accepts conversion cues only inside the detected speech regions.
- Completed tracks are fetched back into the browser; server job files expire automatically and can also be deleted through the API.
- No personal account IDs, local paths, API keys, analytics keys, deployment identifiers, user audio, or model cache files belong in this repository.
- `.env*`, `.openai/hosting.json`, build output, local jobs, and Python environments are ignored.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Licensed under MIT; third-party models retain their own licenses and terms.
