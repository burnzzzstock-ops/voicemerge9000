from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import urljoin, urlparse

import httpx


class ModelStoreError(RuntimeError):
    pass


class ModelSecurityError(ModelStoreError):
    pass


ModelProgress = Callable[[str, int | None, int | None], None]


@dataclass(frozen=True)
class ModelArtifact:
    checkpoint: Path
    index: Path | None
    sha256: str


@dataclass(frozen=True)
class ModelStoreSettings:
    root: Path
    allowed_hosts: tuple[str, ...]
    token_secret: str | None
    max_download_bytes: int
    max_extracted_bytes: int
    max_checkpoint_bytes: int
    max_index_bytes: int

    @classmethod
    def from_environment(cls) -> "ModelStoreSettings":
        hosts = os.getenv(
            "VOICEMERGE_MODEL_HOSTS",
            "huggingface.co,hf.co,drive.google.com,drive.usercontent.google.com,googleusercontent.com,pixeldrain.com,mediafire.com,weights.gg,voice-models.com",
        )
        return cls(
            root=Path(os.getenv("VOICEMERGE_MODEL_CACHE", "/var/lib/voicemerge/models")),
            allowed_hosts=tuple(host.strip().lower() for host in hosts.split(",") if host.strip()),
            token_secret=os.getenv("VOICEMERGE_MODEL_TOKEN_SECRET") or None,
            max_download_bytes=int(os.getenv("VOICEMERGE_MAX_MODEL_DOWNLOAD", str(1024**3))),
            max_extracted_bytes=int(os.getenv("VOICEMERGE_MAX_MODEL_EXTRACTED", str(2 * 1024**3))),
            max_checkpoint_bytes=int(os.getenv("VOICEMERGE_MAX_PTH", str(768 * 1024**2))),
            max_index_bytes=int(os.getenv("VOICEMERGE_MAX_INDEX", str(256 * 1024**2))),
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive_name(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts or ":" in normalized:
        raise ModelSecurityError(f"unsafe archive member: {name!r}")
    return path


def safe_destination(root: Path, name: str) -> Path:
    relative = validate_archive_name(name)
    destination = (root / Path(*relative.parts)).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ModelSecurityError(f"archive member escapes extraction root: {name!r}")
    return destination


class ModelStore:
    def __init__(self, settings: ModelStoreSettings | None = None) -> None:
        self.settings = settings or ModelStoreSettings.from_environment()
        self.settings.root.mkdir(parents=True, exist_ok=True)
        (self.settings.root / "refs").mkdir(exist_ok=True)
        (self.settings.root / "objects").mkdir(exist_ok=True)
        self._lock = threading.Lock()

    def _host_allowed(self, host: str) -> bool:
        host = host.rstrip(".").lower()
        return any(host == allowed or host.endswith(f".{allowed}") for allowed in self.settings.allowed_hosts)

    def _normalize_url(self, value: str) -> str:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if host == "mega.nz" or host.endswith(".mega.nz"):
            raise ModelStoreError("encrypted Mega model links are not yet supported; choose a direct Hugging Face, Drive, Pixeldrain, or weights.gg model")
        if host == "drive.google.com":
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 3 and parts[0] == "file" and parts[1] == "d":
                return f"https://drive.usercontent.google.com/download?id={parts[2]}&export=download&confirm=t"
        if host == "pixeldrain.com":
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) == 2 and parts[0] == "u":
                return f"https://pixeldrain.com/api/file/{parts[1]}"
        return value

    def _validate_remote_url(self, value: str) -> str:
        value = self._normalize_url(value)
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ModelSecurityError("model URL must be an unauthenticated HTTPS URL")
        if not self._host_allowed(parsed.hostname):
            raise ModelSecurityError(f"model host is not trusted: {parsed.hostname}")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)}
        except OSError as error:
            raise ModelStoreError(f"could not resolve model host: {parsed.hostname}") from error
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ModelSecurityError("model URL resolved to a private or reserved address")
        return value

    def _decode_model_id(self, model_id: str) -> tuple[str, str | None]:
        if model_id.startswith("https://"):
            return self._validate_remote_url(model_id), None
        parts = model_id.split(".")
        if len(parts) != 3 or parts[0] != "v1" or not self.settings.token_secret:
            raise ModelSecurityError("model_id must be a trusted HTTPS URL or a signed v1 token")
        try:
            encoded, signature = parts[1], parts[2]
            expected = hmac.new(self.settings.token_secret.encode(), encoded.encode(), hashlib.sha256).digest()
            supplied = base64.urlsafe_b64decode(signature + "===")
            if not hmac.compare_digest(expected, supplied):
                raise ModelSecurityError("invalid model token signature")
            payload = json.loads(base64.urlsafe_b64decode(encoded + "==="))
            if int(payload["exp"]) < int(time.time()):
                raise ModelSecurityError("model token has expired")
            return self._validate_remote_url(str(payload["url"])), payload.get("sha256")
        except ModelSecurityError:
            raise
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ModelSecurityError("invalid model token") from error

    def _download(self, url: str, destination: Path, progress: ModelProgress | None = None) -> None:
        current = self._validate_remote_url(url)
        with httpx.Client(timeout=httpx.Timeout(30, read=300), follow_redirects=False) as client:
            for _ in range(6):
                with client.stream("GET", current, headers={"User-Agent": "VoiceMerge9000/0.2"}) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ModelStoreError("model host returned an empty redirect")
                        current = self._validate_remote_url(urljoin(current, location))
                        continue
                    response.raise_for_status()
                    declared = int(response.headers.get("content-length", "0") or 0)
                    if declared > self.settings.max_download_bytes:
                        raise ModelSecurityError("model package exceeds the download limit")
                    size = 0
                    if progress:
                        progress("downloading_model", 0, declared or None)
                    with destination.open("wb") as target:
                        for chunk in response.iter_bytes(1024 * 1024):
                            size += len(chunk)
                            if size > self.settings.max_download_bytes:
                                raise ModelSecurityError("model package exceeds the download limit")
                            target.write(chunk)
                            if progress:
                                progress("downloading_model", size, declared or None)
                    if progress:
                        progress("extracting_model", size, declared or size)
                    return
            raise ModelSecurityError("model URL redirected too many times")

    def _copy_member(self, source, destination: Path, size: int, limit: int) -> None:
        if size < 1 or size > limit:
            raise ModelSecurityError(f"model file size {size} is outside its allowed range")
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with source, destination.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                written += len(chunk)
                if written > size or written > limit:
                    raise ModelSecurityError("archive member exceeded its declared or allowed size")
                target.write(chunk)

    def _extract_zip(self, package: Path, root: Path) -> None:
        total = 0
        selected = 0
        with zipfile.ZipFile(package) as archive:
            for member in archive.infolist():
                validate_archive_name(member.filename)
                if member.is_dir():
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ModelSecurityError("symbolic links are not allowed in model archives")
                suffix = Path(member.filename).suffix.lower()
                if suffix not in {".pth", ".index"}:
                    continue
                limit = self.settings.max_checkpoint_bytes if suffix == ".pth" else self.settings.max_index_bytes
                if suffix == ".index" and not 1 <= member.file_size <= limit:
                    continue
                total += member.file_size
                if total > self.settings.max_extracted_bytes:
                    raise ModelSecurityError("extracted model files exceed the limit")
                self._copy_member(archive.open(member), safe_destination(root, member.filename), member.file_size, limit)
                selected += 1
        if not selected:
            raise ModelStoreError("archive contains no .pth or .index model files")

    def _extract_tar(self, package: Path, root: Path) -> None:
        total = 0
        selected = 0
        with tarfile.open(package, "r:*") as archive:
            for member in archive.getmembers():
                validate_archive_name(member.name)
                if member.issym() or member.islnk():
                    raise ModelSecurityError("links are not allowed in model archives")
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lower()
                if suffix not in {".pth", ".index"}:
                    continue
                limit = self.settings.max_checkpoint_bytes if suffix == ".pth" else self.settings.max_index_bytes
                if suffix == ".index" and not 1 <= member.size <= limit:
                    continue
                total += member.size
                if total > self.settings.max_extracted_bytes:
                    raise ModelSecurityError("extracted model files exceed the limit")
                source = archive.extractfile(member)
                if source is None:
                    raise ModelStoreError(f"could not read {member.name}")
                self._copy_member(source, safe_destination(root, member.name), member.size, limit)
                selected += 1
        if not selected:
            raise ModelStoreError("archive contains no .pth or .index model files")

    def _validate_checkpoint(self, checkpoint: Path) -> None:
        environment = os.environ.copy()
        environment["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        command = [
            os.getenv("VOICEMERGE_RVC_PYTHON", sys.executable),
            "-c",
            "import sys,torch; torch.load(sys.argv[1], map_location='cpu', weights_only=True)",
            str(checkpoint),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=environment, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ModelSecurityError("checkpoint safety validation could not complete") from error
        if result.returncode:
            detail = (result.stderr or result.stdout).strip().splitlines()[-1:]
            raise ModelSecurityError(f"checkpoint failed weights-only validation{': ' + detail[0] if detail else ''}")

    def resolve(self, model_id: str, progress: ModelProgress | None = None) -> ModelArtifact:
        url, expected_hash = self._decode_model_id(model_id)
        reference_key = hashlib.sha256(url.encode()).hexdigest()
        reference = self.settings.root / "refs" / f"{reference_key}.json"
        with self._lock:
            if reference.exists():
                metadata = json.loads(reference.read_text(encoding="utf-8"))
                object_root = self.settings.root / "objects" / metadata["sha256"]
                checkpoint = object_root / metadata["checkpoint"]
                index = object_root / metadata["index"] if metadata.get("index") else None
                if checkpoint.is_file() and (index is None or index.is_file()):
                    if progress:
                        progress("model_ready", None, None)
                    return ModelArtifact(checkpoint, index, metadata["sha256"])

            staging = self.settings.root / f".staging-{reference_key}"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            package = staging / "download"
            try:
                if progress:
                    progress("resolving_model", None, None)
                self._download(url, package, progress)
                package_hash = sha256_file(package)
                if expected_hash and not hmac.compare_digest(package_hash, expected_hash.lower()):
                    raise ModelSecurityError("model package checksum did not match its signed token")
                extracted = staging / "files"
                extracted.mkdir()
                url_suffix = Path(urlparse(url).path).suffix.lower()
                if url_suffix == ".pth":
                    if package.stat().st_size > self.settings.max_checkpoint_bytes:
                        raise ModelSecurityError("checkpoint exceeds the size limit")
                    shutil.move(package, extracted / "model.pth")
                elif url_suffix == ".index":
                    raise ModelStoreError("a model_id must point to a checkpoint or model archive, not only an index")
                elif zipfile.is_zipfile(package):
                    self._extract_zip(package, extracted)
                elif tarfile.is_tarfile(package):
                    self._extract_tar(package, extracted)
                else:
                    raise ModelStoreError("model package must be a .pth, .zip, .tar, .tar.gz, or .tgz file")

                checkpoints = sorted(extracted.rglob("*.pth"), key=lambda path: path.stat().st_size, reverse=True)
                indexes = sorted(extracted.rglob("*.index"), key=lambda path: ("added_" not in path.name, -path.stat().st_size))
                if not checkpoints:
                    raise ModelStoreError("model package contains no checkpoint")
                checkpoint = checkpoints[0]
                if progress:
                    progress("validating_model", None, None)
                self._validate_checkpoint(checkpoint)
                object_root = self.settings.root / "objects" / package_hash
                if not object_root.exists():
                    shutil.move(extracted, object_root)
                selected_checkpoint = object_root / checkpoint.relative_to(extracted)
                selected_index = object_root / indexes[0].relative_to(extracted) if indexes else None
                metadata = {
                    "sha256": package_hash,
                    "checkpoint": str(selected_checkpoint.relative_to(object_root)),
                    "index": str(selected_index.relative_to(object_root)) if selected_index else None,
                }
                reference.write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
                if progress:
                    progress("model_ready", None, None)
                return ModelArtifact(selected_checkpoint, selected_index, package_hash)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
