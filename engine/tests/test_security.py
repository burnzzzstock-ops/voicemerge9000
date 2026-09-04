from __future__ import annotations

import hashlib
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
