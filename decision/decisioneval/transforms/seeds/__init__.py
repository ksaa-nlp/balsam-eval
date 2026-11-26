"""Hand-written transforms for the protocols every native adapter speaks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

SEEDS_DIR = Path(__file__).resolve().parent
SEED_NAMES = ("systemone", "openai-decisions")


def seed_transforms(name: str) -> tuple[str, str]:
    """The (request, response) JSONata for one seeded protocol."""
    if name not in SEED_NAMES:
        raise KeyError(f"no seeded protocol named {name!r}")
    return tuple(  # type: ignore[return-value]
        (SEEDS_DIR / f"{name}.{side}.jsonata").read_text(encoding="utf-8") for side in ("request", "response")
    )


def request_input(
    *, model: str, instruction: str, input_text: str, options: Sequence[str], option_ids: Sequence[str]
) -> dict[str, Any]:
    """What a request transform sees for one question (plan §6.3). Never the key."""
    return {
        "model": model,
        "instruction": instruction,
        "input": input_text,
        "state": f"{instruction}\n\n{input_text}".strip(),
        "options": list(options),
        "option_ids": list(option_ids),
        "criteria": dict(zip(option_ids, options, strict=True)),
    }
