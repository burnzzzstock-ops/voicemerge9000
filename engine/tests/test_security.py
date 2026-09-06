from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import zipfile
from pathlib import Path

import pytest

from engine.model_store import (
    ModelSecurityError,
    ModelStore,
    ModelStoreSettings,
    safe_destination,
    sha256_file,
    validate_archive_name,
)


@pytest.mark.parametrize("name", ["../model.pth", "folder/../../model.pth", "/model.pth", "C:/model.pth", "C:\\model.pth"])
def test_archive_traversal_is_rejected(name: str) -> None:
    with pytest.raises(ModelSecurityError):
        validate_archive_name(name)


def test_safe_destination_stays_inside_root(tmp_path: Path) -> None:
    destination = safe_destination(tmp_path, "nested/model.pth")
    assert tmp_path.resolve() in destination.parents


def test_streamed_sha256(tmp_path: Path) -> None:
    payload = b"voice-model-test" * 1000
    path = tmp_path / "model.zip"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_zip_traversal_is_rejected_before_extraction(tmp_path: Path) -> None:
    package = tmp_path / "bad.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../outside.pth", b"unsafe")
    settings = ModelStoreSettings(
        root=tmp_path / "cache",
        allowed_hosts=("huggingface.co",),
        token_secret=None,
        max_download_bytes=1024,
        max_extracted_bytes=1024,
        max_checkpoint_bytes=1024,
        max_index_bytes=1024,
    )
    store = ModelStore(settings)
    with pytest.raises(ModelSecurityError):
        store._extract_zip(package, tmp_path / "extract")


def test_oversized_optional_index_is_skipped(tmp_path: Path) -> None:
    package = tmp_path / "large-index.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("voice.pth", b"checkpoint")
        archive.writestr("voice.index", b"index-is-optional")
    settings = ModelStoreSettings(
        root=tmp_path / "cache",
        allowed_hosts=("huggingface.co",),
        token_secret=None,
        max_download_bytes=1024,
        max_extracted_bytes=1024,
        max_checkpoint_bytes=1024,
        max_index_bytes=4,
    )
    store = ModelStore(settings)
    extracted = tmp_path / "extract"

    store._extract_zip(package, extracted)

    assert (extracted / "voice.pth").read_bytes() == b"checkpoint"
    assert not (extracted / "voice.index").exists()


def test_oversized_checkpoint_is_still_rejected(tmp_path: Path) -> None:
    package = tmp_path / "large-checkpoint.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("voice.pth", b"checkpoint")
    settings = ModelStoreSettings(
        root=tmp_path / "cache",
        allowed_hosts=("huggingface.co",),
        token_secret=None,
        max_download_bytes=1024,
        max_extracted_bytes=1024,
        max_checkpoint_bytes=4,
        max_index_bytes=1024,
    )
    store = ModelStore(settings)

    with pytest.raises(ModelSecurityError):
        store._extract_zip(package, tmp_path / "extract")


def test_untrusted_or_insecure_model_hosts_are_rejected(tmp_path: Path) -> None:
    settings = ModelStoreSettings(
        root=tmp_path / "cache",
        allowed_hosts=("huggingface.co",),
        token_secret=None,
        max_download_bytes=1024,
        max_extracted_bytes=1024,
        max_checkpoint_bytes=1024,
        max_index_bytes=1024,
    )
    store = ModelStore(settings)
    with pytest.raises(ModelSecurityError):
        store._validate_remote_url("http://huggingface.co/model.zip")
    with pytest.raises(ModelSecurityError):
        store._validate_remote_url("https://example.com/model.zip")


def test_cached_model_still_enforces_signed_checksum(tmp_path: Path, monkeypatch) -> None:
    settings = ModelStoreSettings(root=tmp_path / "cache", allowed_hosts=("huggingface.co",),
        token_secret="unit-test-only", max_download_bytes=1024, max_extracted_bytes=1024,
        max_checkpoint_bytes=1024, max_index_bytes=1024)
    store = ModelStore(settings)
    url = "https://huggingface.co/example/model.zip"
    digest = "a" * 64
    obj = settings.root / "objects" / digest
    obj.mkdir()
    checkpoint = obj / "model.pth"
    checkpoint.write_bytes(b"previously-validated-test-checkpoint")
    reference = settings.root / "refs" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    reference.write_text(json.dumps({"sha256": digest, "checkpoint": "model.pth", "index": None}))
    monkeypatch.setattr(store, "_validate_remote_url", lambda value: value)
    monkeypatch.setattr(store, "_download", lambda *_: pytest.fail("cache lookup must not download"))
    def token(expected_hash: str) -> str:
        payload = json.dumps({"url": url, "exp": int(time.time()) + 60, "sha256": expected_hash})
        encoded = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        signature = hmac.new(settings.token_secret.encode(), encoded.encode(), hashlib.sha256).digest()
        return "v1." + encoded + "." + base64.urlsafe_b64encode(signature).decode().rstrip("=")
    assert store.resolve(token(digest.upper())).checkpoint == checkpoint
    with pytest.raises(ModelSecurityError, match="cached model package checksum"):
        store.resolve(token("b" * 64))
    assert checkpoint.read_bytes() == b"previously-validated-test-checkpoint"
