"""Download, normalise and validate a job's pool files; non-MCQ pools are skipped with a reason."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from decisioneval.datasets import DatasetError, load_dataset
from decisioneval.remote.normalize import NotMcqPool, PoolDocument, normalize_pool
from decisioneval.remote.storage import ObjectStore


@dataclass(frozen=True)
class SkippedPool:
    source: str
    reason: str


@dataclass(frozen=True)
class PreparedPools:
    directory: Path
    pools: tuple[PoolDocument, ...]
    skipped: tuple[SkippedPool, ...]


def slug_for(source: str) -> str:
    """Basename plus a path hash: basenames collide across categories. Same as balsam-eval."""
    stem = Path(source).stem or "pool-file"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def prepare_pools(store: ObjectStore, sources: tuple[str, ...], directory: Path) -> PreparedPools:
    """Download, normalise and validate every pool. Storage errors propagate."""
    directory.mkdir(parents=True, exist_ok=True)
    pools: list[PoolDocument] = []
    skipped: list[SkippedPool] = []
    for source in sources:
        try:
            raw = json.loads(store.read_bytes(source).decode("utf-8-sig"))
            pool = normalize_pool(raw, source)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            skipped.append(SkippedPool(source, f"not valid JSON: {exc}"))
            continue
        except NotMcqPool as exc:
            skipped.append(SkippedPool(source, str(exc)))
            continue

        path = directory / f"{slug_for(source)}.json"
        path.write_text(json.dumps(pool.document, ensure_ascii=False), encoding="utf-8")
        reason = _unscorable(path)
        if reason:
            path.unlink()
            skipped.append(SkippedPool(source, reason))
            continue
        pools.append(pool)
    return PreparedPools(directory=directory, pools=tuple(pools), skipped=tuple(skipped))


def _unscorable(path: Path) -> str | None:
    try:
        dataset = load_dataset(path)
    except DatasetError as exc:
        return str(exc)
    if not dataset.questions:
        return "no scorable questions"
    return None
