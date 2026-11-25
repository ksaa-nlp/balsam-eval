"""Object storage as read/write/list; GCS is imported lazily, so a local run never loads it."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol


class StorageError(Exception):
    """An object could not be read, written or listed."""


class ObjectStore(Protocol):
    def read_bytes(self, path: str) -> bytes: ...

    def write_bytes(self, path: str, data: bytes, *, if_absent: bool = False) -> bool:
        """Write `data`. With `if_absent`, return False instead of overwriting."""
        ...

    def list(self, prefix: str) -> list[str]: ...


def join(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name.lstrip('/')}" if prefix else name


class LocalStore:
    """A directory standing in for a bucket. Paths cannot escape `root`."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if target != self.root and self.root not in target.parents:
            raise StorageError(f"path {path!r} escapes the storage root")
        return target

    def read_bytes(self, path: str) -> bytes:
        try:
            return self._resolve(path).read_bytes()
        except OSError as exc:
            raise StorageError(f"could not read {path}: {exc}") from exc

    def write_bytes(self, path: str, data: bytes, *, if_absent: bool = False) -> bool:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if if_absent:
            try:
                with target.open("xb") as handle:
                    handle.write(data)
            except FileExistsError:
                return False
            except OSError as exc:
                raise StorageError(f"could not write {path}: {exc}") from exc
            return True
        handle = tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".tmp")
        try:
            with handle as stream:
                stream.write(data)
            os.replace(handle.name, target)
        except OSError as exc:
            Path(handle.name).unlink(missing_ok=True)
            raise StorageError(f"could not write {path}: {exc}") from exc
        return True

    def list(self, prefix: str) -> list[str]:
        base = self._resolve(prefix) if prefix else self.root
        if not base.is_dir():
            return []
        return sorted(
            str(path.relative_to(self.root))
            for path in base.rglob("*")
            if path.is_file() and not path.name.endswith(".tmp")
        )


class GcsStore:
    """One bucket, through google-cloud-storage and application default credentials."""

    def __init__(self, bucket: str, client=None):
        self.bucket_name = bucket
        self._client = client

    def _bucket(self):
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise StorageError(
                    "google-cloud-storage is not installed; run `uv sync` in decision/"
                ) from exc
            self._client = storage.Client()
        return self._client.bucket(self.bucket_name)

    def read_bytes(self, path: str) -> bytes:
        try:
            return self._bucket().blob(path).download_as_bytes()
        except Exception as exc:
            raise StorageError(f"could not read gs://{self.bucket_name}/{path}: {exc}") from exc

    def write_bytes(self, path: str, data: bytes, *, if_absent: bool = False) -> bool:
        blob = self._bucket().blob(path)
        try:
            blob.upload_from_string(
                data,
                content_type="application/json",
                if_generation_match=0 if if_absent else None,
            )
        except Exception as exc:
            if if_absent and _is_precondition_failure(exc):
                return False
            raise StorageError(f"could not write gs://{self.bucket_name}/{path}: {exc}") from exc
        return True

    def list(self, prefix: str) -> list[str]:
        try:
            return sorted(blob.name for blob in self._bucket().list_blobs(prefix=prefix))
        except Exception as exc:
            raise StorageError(f"could not list gs://{self.bucket_name}/{prefix}: {exc}") from exc


def _is_precondition_failure(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 412 or type(exc).__name__ == "PreconditionFailed"
