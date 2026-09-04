# VoiceMerge9000

VoiceMerge9000 is a one-screen character voice casting desk for scenes, songs, and dialogue. Upload one audio file, review who speaks where, assign a different RVC model to each speaker, then click **Convert all + merge**. The app converts only the reviewed cues, rebuilds full-length speaker stems at their original timestamps, and mixes them against the original scene.

Use fictional, stylized, public-domain, or otherwise authorized voices. The project does not bundle voice models and must not be used to impersonate a real person without permission.

## What is implemented

- Browser-local speech-activity detection and an editable acoustic speaker draft
- One to six independent Voice A–F character assignments
- In-app `voice-models.com` search, samples, metadata, and direct model selection
- A same-origin job API: the user never has to open a second converter
- A typed FastAPI service with upload, progress, per-speaker tracks, and deletion endpoints
- Official RVC CLI inference, CUDA detection with CPU fallback, and one model load per speaker batch
- Sample-accurate cue extraction, duration-drift correction, 5 ms edge fades, silent full-length stems, and a browser-local 48 kHz / 24-bit master
- Failure isolation: one failed speaker does not discard successful tracks
- Model cache and trust boundary: HTTPS allowlist or HMAC-signed IDs, redirect/DNS checks, size ceilings, safe archive extraction, SHA-256 object storage, and PyTorch weights-only preflight
- A development copy backend so the entire upload → poll → download → merge flow can be tested without a GPU or model

The speaker draft is a guess, not full diarization. Music, effects, crosstalk, and similar voices can confuse it; human review is intentionally part of the workflow.

## Easiest local run

Requirements: Docker, Docker Compose, and enough disk space for the RVC runtime and model cache.

1. Copy `.env.example` to `.env` and replace both `change-me` values with the same long random token.
2. Start the app:

```bash
docker compose up --build
```

3. Open `http://localhost:3000`. Only the VoiceMerge interface is exposed; the Python worker stays on Docker's private network.

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

Windows PowerShell uses `$env:NAME='value'` before each command. Open `http://localhost:3000`.

Tests and frontend checks:

```bash
.venv/bin/pytest engine/tests
pnpm exec tsc --noEmit
pnpm run lint
pnpm build
```

## Engine API

- `GET /api/v1/health`
- `POST /api/v1/jobs` as multipart form data with `audio_file` and JSON `job_data`
- `GET /api/v1/jobs/{job_id}` for progress and track URLs
- `GET /api/v1/jobs/{job_id}/tracks/{speaker_id}.wav`
- `DELETE /api/v1/jobs/{job_id}` after the tracks are downloaded

The browser calls `/api/engine/*`; the web app proxies that to `RVC_ENGINE_URL` and injects the private engine token. Set `VOICEMERGE_MODEL_HOSTS` to the smallest host allowlist you need. Production model registries can issue `v1.<payload>.<hmac>` model IDs containing an expiry, URL, and optional SHA-256.

## Deployment

The web interface can run on Sites or any compatible JavaScript host. RVC requires Python, FFmpeg, large model files, and usually CUDA, so deploy `engine/Dockerfile` to a GPU/container host and set the web deployment's private `RVC_ENGINE_URL` and `RVC_ENGINE_API_TOKEN`. This still presents one app and one button to the user; the inference worker is an internal service.

## Privacy

- Speaker analysis happens in the browser.
- Source audio reaches only the configured VoiceMerge engine after **Convert all + merge** is clicked.
- Completed tracks are fetched back into the browser; server job files expire and can be deleted immediately.
- No personal account IDs, local paths, API keys, analytics keys, deployment identifiers, user audio, or model cache files belong in this repository.
- `.env*`, `.openai/hosting.json`, build output, local jobs, and Python environments are ignored.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Licensed under MIT; third-party models retain their own licenses and terms.
