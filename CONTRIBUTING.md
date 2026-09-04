# Contributing

Thanks for helping make VoiceMerge9000 easier and safer for creators.

## Ground rules

- Do not commit voice-model binaries, copyrighted source clips, personal data, tokens, local paths, or deployment/account identifiers.
- Use fictional, stylized, public-domain, or authorized voice examples in tests and documentation.
- Keep provider integrations behind small adapters so the core workflow remains portable.
- Label heuristic or experimental output honestly. Speaker guesses always need human review.
- Prefer browser-local or local-companion processing. Any external upload must be explicit in the interface.

## Pull requests

1. Create a focused branch.
2. Run `pnpm build` and lint the files you changed.
3. Explain the creator-facing behavior, privacy impact, and manual test used.
4. Do not include generated audio or model archives in the pull request.
