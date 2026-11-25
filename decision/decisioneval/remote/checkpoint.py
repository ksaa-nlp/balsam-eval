"""Scored items saved as they finish, so a relaunched build resumes instead of starting over.

Chunks are create-only, under the adapter's content hash: a different adapter
version never resumes another's items, and a racing writer can't overwrite one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from decisioneval.remote.storage import ObjectStore, join
from decisioneval.store import redact_secrets, scrub_key

CHUNK_ITEMS = 10
CHUNK_SUFFIX = ".jsonl"


class Checkpoint:
    def __init__(
        self, store: ObjectStore, results_path: str, content_hash: str, target: str = "", key: str | None = None
    ):
        self.store = store
        self.key = key
        # Models on one seeded protocol share a hash, so the endpoint and model scope the chunks too.
        scope = hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
        self.prefix = join(results_path, f"_checkpoint/{content_hash}/{scope}")
        self._pending: list[dict[str, Any]] = []
        self._next_seq: int | None = None

    def load(self) -> list[dict[str, Any]]:
        """Every saved item, in the order saved, each item once."""
        seen: set[tuple[str, str]] = set()
        items: list[dict[str, Any]] = []
        for path in self._chunks():
            for line in self.store.read_bytes(path).decode("utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = item_key(record)
                if key not in seen:
                    seen.add(key)
                    items.append(record)
        return items

    def add(self, item: dict[str, Any]) -> None:
        self._pending.append(item)
        if len(self._pending) >= CHUNK_ITEMS:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        if self._next_seq is None:
            self._next_seq = len(self._chunks())
        data = "".join(
            json.dumps(scrub_key(redact_secrets(item), self.key), ensure_ascii=False) + "\n" for item in self._pending
        ).encode("utf-8")
        # Another writer took this sequence number: move past it rather than overwrite.
        while not self.store.write_bytes(join(self.prefix, f"{self._next_seq:06d}{CHUNK_SUFFIX}"), data, if_absent=True):
            self._next_seq += 1
        self._next_seq += 1
        self._pending = []

    def _chunks(self) -> list[str]:
        return sorted(path for path in self.store.list(self.prefix) if path.endswith(CHUNK_SUFFIX))


def item_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("dataset")), str(record.get("item_id")))
