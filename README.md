# VoiceMerge9000

VoiceMerge9000 is a browser-first character voice casting desk for scenes, songs, and dialogue. Upload one audio file, review a local speaker-section draft, assign community RVC character models, prepare timed conversion jobs, and merge the returned voices on the original timeline.

The project is designed for fictional, stylized, public-domain, or otherwise authorized voices. It does not bundle voice models and it does not support impersonating a real person without their permission.

## What works now

- Browser-local speech-activity detection and a lightweight acoustic speaker draft
- A human review list with section playback, reassignment, and playhead splitting
- In-app search of the public `voice-models.com` catalog, including samples and model metadata
- Independent Voice A–F character assignments
- Full-length, time-locked WAV job exports for each detected speaker
- Deep links that load the chosen model in the external converter
- Return-file upload and a browser-local 48 kHz / 24-bit WAV scene merge
- Simple natural-language edit commands such as `swap A and B`, `make B louder`, and `use 4 speakers`

## Important current limitation

RVC inference is not yet bundled into the web app. VoiceMerge9000 prepares the timed audio jobs and chosen model links, but conversion currently happens in the external converter before the files are returned for the final merge. The first major community milestone is a local companion engine so the whole conversion queue can run with one click and no audio upload.

The current acoustic speaker draft is intentionally labeled as a guess. It is not full speaker diarization and can be confused by music, sound effects, crosstalk, or very similar voices. Human review is part of the workflow.

## Run locally

Requirements: Node.js 22.13+ and pnpm.

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`.

```bash
pnpm build
```

## Privacy

- Uploaded audio is decoded and analyzed in the browser.
- Audio leaves the browser only when the user deliberately sends a prepared job to an external conversion service.
- No personal account IDs, local paths, API keys, analytics keys, or deployment identifiers belong in this repository.
- `.env*` and the local Sites deployment binding are ignored by Git.

## Roadmap

1. Replace the lightweight acoustic clustering with a robust local speaker-diarization adapter.
2. Add a local RVC engine with model caching, checksum validation, job progress, and cancellation.
3. Add speech/background stem separation so dialogue can be replaced while music and effects remain clean.
4. Add editable fades, overlaps, pitch controls, and per-character conversion presets.
5. Save and reopen portable project manifests without bundling copyrighted model files.

## Contributing

Issues and pull requests are welcome. Keep contributions provider-neutral, privacy-preserving, and understandable to non-technical creators. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

MIT. Third-party voice models, samples, and conversion services retain their own licenses and terms.
