"""Fixtures: a real MCQ dataset on disk, and fake models that answer it.

Nothing here touches the network. A "model" is a callable with the HTTP
transport's signature that builds a reply from the question it was asked.
"""

from __future__ import annotations



from decisioneval.spec.transport import Reply

OPTIONS_3 = ["نعم", "لا", "ربما"]
OPTIONS_4 = ["شرق", "غرب", "شمال", "جنوب"]


def mcq_document(name: str, options: list[str], count: int = 6) -> dict:
    """A dataset in the loader's format, gold rotating through the options."""
    return {
        "Type of result": "MCQ",
        "name": name,
        "category": "Factuality",
        "task": name,
        "data": {
            "test": [
                {
                    "id": f"{name}-{index}",
                    "instruction": "اختر الإجابة الصحيحة",
                    "input": [f"نص السؤال {index}"],
                    "mcq": list(options),
                    "output": options[index % len(options)],
                }
                for index in range(count)
            ]
        },
    }


def vector(count: int, pick: int) -> list[float]:
    """A deterministic distribution that sums to 1."""
    rest = round(0.3 / (count - 1), 6) if count > 1 else 0.0
    values = [rest] * count
    values[pick] = round(1.0 - rest * (count - 1), 6)
    return values


class FakeModel:
    """A model endpoint. `shape` decides what envelope it answers in."""

    def __init__(self, shape: str = "system_one", pick=None, fail_after: int | None = None):
        self.shape = shape
        self.pick = pick or (lambda question: 0)
        self.fail_after = fail_after
        self.calls = 0

    def __call__(self, url, body, headers, timeout_s, method="POST"):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            return Reply(status=503, payload={"error": "unavailable"}, latency_ms=1.0, headers={})
        criteria = body["questions"]["answer"]["criteria"]
        ids = list(criteria)
        texts = [criteria[key] for key in ids]
        index = self.pick(body) if callable(self.pick) else 0
        index = index % len(ids)
        return Reply(
            status=200,
            payload=self.payload(ids, texts, index),
            latency_ms=4.0,
            headers={},
        )

    def payload(self, ids: list[str], texts: list[str], pick: int):
        values = vector(len(ids), pick)
        if self.shape == "system_one":
            return {
                "answers": {
                    "answer": {
                        "type": "choice",
                        "choice": ids[pick],
                        "probabilities": dict(zip(ids, values)),
                        "confidence": values[pick],
                    }
                }
            }
        if self.shape == "index_keyed":
            return {
                "out": {
                    "pick": pick,
                    "p": {str(position): values[position] for position in range(len(ids))},
                }
            }
        if self.shape == "ranked_pairs":
            rows = sorted(
                ({"label": texts[position], "score": values[position]} for position in range(len(ids))),
                key=lambda row: -row["score"],
            )
            return {"best": texts[pick], "ranking": rows}
        if self.shape == "logits_only":
            return {
                "argmax": ids[pick],
                "logits": {ids[position]: (4.0 if position == pick else -1.5) for position in range(len(ids))},
            }
        if self.shape == "ambiguous":
            # two different vectors that both sum to 1 on the option ids
            other = vector(len(ids), (pick + 1) % len(ids))
            return {
                "pick": ids[pick],
                "p": dict(zip(ids, values)),
                "q": dict(zip(ids, other)),
            }
        if self.shape == "opaque":
            # nothing resembling a choice: not even a degraded mapping exists
            return {"status": "ok", "latency": 12, "note": "see dashboard"}
        raise AssertionError(f"unknown fake shape {self.shape!r}")


REQUEST = {
    "transport": "http",
    "url": "https://fake.invalid/v1/decide",
    "body": {
        "state": "{{state}}",
        "questions": {"answer": {"type": "choice", "criteria": "{{criteria}}"}},
    },
}
