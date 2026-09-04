# Security and privacy

Please report a security or privacy issue with GitHub's private vulnerability reporting feature instead of opening a public issue.

Do not include private audio, voice models, access tokens, personal information, or local filesystem paths in a report. Use a minimal synthetic reproduction.

The hosted interface treats model-catalog entries as untrusted community metadata. Model packages are not executed by the web app. A future local engine must verify archive type, size limits, checksums when available, extraction paths, and inference inputs before use.
