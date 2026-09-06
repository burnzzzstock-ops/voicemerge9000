# VoiceMerge9000 contributor handoff

## Status and purpose

This is an unfinished local voice-swapping application being handed off for community development. The intended experience is: upload media, review vocal sections and speaker guesses, choose character models, merge, preview, adjust, and download. No external inference website is required. Natural-language requests such as “put these three characters in this song” are a future workflow, not a completed feature.

Real multi-character conversion has run locally through Demucs, Silero, RVC, and a 48 kHz stereo master. User listening feedback still reports missing parts, robotic artifacts, inconsistent balance, and weak likeness for some models. Passing timing and peak tests does not establish acceptable sound quality.

## Start here

1. Follow the Docker or development setup in [README.md](README.md). Docker is the reference real-inference environment; the development copy backend does not convert voices.
2. Run the checks in [CONTRIBUTING.md](CONTRIBUTING.md).
3. Pick one issue below, reproduce it with synthetic or permissioned media, and submit a focused change with before/after evidence.
4. Keep all downloaded checkpoints, source media, generated audio, account configuration, and private diagnostics out of commits.

## Architecture map

| Area | Entry points | Responsibility |
| --- | --- | --- |
| Interface | `app/voice-cast-studio.tsx` | Upload, timeline review, casting, progress, playback |
| Frontend safety | `lib/operation.ts`, `lib/timing.ts` | Stale operation guards and editable cue bounds |
| Server proxy | `app/api/engine/[...path]/route.ts` | Same-origin authenticated engine requests |
| Model discovery | `app/api/models/search/route.ts` | Community catalog adapter; treat metadata as untrusted |
| Jobs | `engine/api.py`, `engine/schemas.py` | Analysis, atomic conversion reservation, attempts, tracks, retries |
| Isolation | `engine/separator.py`, `engine/vad.py` | Stereo background separation and vocal activity detection |
| Conversion | `engine/rvc_engine.py`, `engine/rvc_worker.py` | Isolated vocal cue inference via pinned upstream RVC |
| Audio assembly | `engine/audio.py` | Resampling, exact placement, padding/trimming, gain and peak control |
| Safety | `engine/model_store.py`, `engine/gpu_budget.py` | Checkpoint/download validation and GPU working budget |

The frontend uses React with Next-style routing through **vinext/Vite**, not the standard Next.js production server. The production local web command uses Wrangler. The engine is FastAPI with separate API and RVC Python environments in Docker. Job metadata lives in memory; restart recovery is not durable.

## Priority 1: stop losing vocal material

**Problem:** The final mix uses separated background plus selected converted cues. Vocal material outside the selected regions is omitted. Speech VAD can reject singing, quiet syllables, and short interjections. Current boundary editing is constrained to existing detections, so it cannot recover an entirely missed phrase.

**Next work:** Add a reviewable vocal-coverage view and safe insertion of missed regions from isolated vocals. Design an explicit choice for preserving unconverted original vocals; never silently label fallback audio fully converted. Investigate a song-specific analysis mode rather than treating speech VAD as permission to delete everything else.

**Acceptance:** A synthetic isolated vocal event outside VAD remains visibly unresolved until the user chooses how to handle it. Tests cover short words, sustained singing, manual missing-region insertion, boundary edits, and no accidental double dialogue. RVC must still never receive the mixed soundtrack.

## Priority 2: verify RVC index use and per-voice quality

**Problem:** The neutral index alias can fall back to a symlink across Docker mounts. Upstream resolves that symlink, then rewrites `trained_` to `added_`, which can disable retrieval despite a successful conversion. One inspected trained index contained no vectors; fixing the alias alone would not make it useful. Existing subprocess mocks do not reproduce this upstream behavior.

**Next work:** Use a verified neutral regular-file path where needed, validate index dimensions/vector availability, and surface whether retrieval actually ran. Keep archive size limits and checkpoint safety intact. Add per-character settings and short auditions; the current global settings are not suitable evidence of a model's best achievable likeness.

**Acceptance:** Exercise the real upstream path handling across separate mounts; test empty, incompatible, and valid indexes. A missing or unusable index must be clearly reported. Do not silently raise download limits or relax `weights_only=True` to make a model load.

## Priority 3: protect words and improve mixing

**Problem:** Arbitrary output length mismatch is reconciled by padding/trimming without a content-fidelity gate. Exact duration can hide a missing word. Whole-speaker normalization does not preserve local vocal dynamics. Mono converted vocals do not recreate the original vocal stereo placement, even when the background is preserved.

**Next work:** Investigate bounded context around dry vocal cues and explicit excessive-length/coverage warnings. Compare source-relative vocal levels over time, smooth gain transitions, and audition transitions before changing defaults. Do not assume low sample peaks explain or exclude perceived cutoffs.

**Acceptance:** Regression tests cover onset/end preservation, excessive mismatches, fades, stereo background, loudness, and oversampled peaks. Include short permissioned listening comparisons. Do not use waveform stretching or `numpy.interp` for alignment. Preserve exact scene duration and 48 kHz output.

## Priority 4: reliable agent workflow and reproducible releases

- Build a tool-facing workflow over the existing job API: analyze, propose a cast/coverage plan, review uncertainty, convert, inspect failures, retry, and return an addressable result. The existing browser model-assignment tool is not a full autonomous workflow.
- Add durable job recovery and document cleanup after restarts. Keep attempt downloads immutable and old responses unable to overwrite current results.
- Keep the service loopback-only by default. Public/remote access needs a separate authentication, authorization, upload-limit, and abuse-control design.
- Pin remaining mutable model-weight revisions and audit dependency/weight redistribution terms before publishing container images. The Dockerfile pins RVC source, but some weight downloads still use `main`.
- Add clean-checkout CI with synthetic tests; keep actual inference verification separate and explicitly report required weights/hardware. Do not let copy mode masquerade as real inference.

## Verification record and limitations

During prior local development, 42 Python tests and 5 frontend regression tests passed, along with lint, type checking, production builds, browser workflow checks, and real multi-character CPU/GPU conversion. These are historical checks, not a CI guarantee for every checkout. Independent review addressed several timing, retry, concurrency, and security problems; it did not certify audio quality, and later index/coverage problems remain above.

GPU measurements on an 8 GB laptop GPU stayed below the configured 6.5 GiB working budget in tested runs. This is not a universal guarantee for other inputs, models, hardware, or concurrently running applications. Separation bleed, overlapping speakers, singing detection, vocal stereo placement, and model quality remain limitations. No private test recordings or generated masters are included with this handoff.

## Publication checklist for the maintainer

- Review the actual diff; do not blanket-stage unrelated untracked files.
- Include required new source files: `engine/gpu_budget.py`, `engine/rvc_worker.py`, `engine/tests/test_regressions.py`, `lib/operation.ts`, `lib/timing.ts`, and the two `tests/*.test.mjs` regressions.
- Run checks from a clean checkout or source-only staging copy, not just the development directory.
- Inspect tracked files and commit metadata for secrets, account identifiers, private paths, media, and models. Ignore rules do not remove already tracked files or old history.
- Keep the existing MIT license for project code; separately review upstream dependencies, downloaded weights, community models, and media rights. This handoff is not a license audit of all dependencies.
- Verify private vulnerability reporting is enabled before advertising it as available.
- Obtain owner approval before pushing, publishing, changing repository visibility, or rewriting history. Public account attribution and existing history may remain discoverable; anonymity is not promised.
