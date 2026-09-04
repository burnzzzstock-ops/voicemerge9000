# Security and privacy

Please report a security or privacy issue with GitHub's private vulnerability reporting feature instead of opening a public issue.

Do not include private audio, voice models, access tokens, personal information, or local filesystem paths in a report. Use a minimal synthetic reproduction.

The hosted interface treats model-catalog entries as untrusted community metadata. The engine downloads models only from its configured HTTPS host allowlist; signed model tokens can add expiry and checksum enforcement without bypassing that list. It limits compressed and extracted sizes, rejects archive traversal and links, caches packages by SHA-256, and preflights checkpoints with PyTorch weights-only loading. The RVC subprocess also forces weights-only loading.

Do not expose the engine directly to the public internet without an API token, TLS, request-size limits, and host-level rate limiting. Keep `RVC_ENGINE_API_TOKEN`, `VOICEMERGE_API_TOKEN`, deployment identifiers, model caches, and user audio out of Git.
