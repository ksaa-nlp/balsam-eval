"""Canary suite v1: does an adapter actually deliver the question to the model?

24 synthetic Arabic general-knowledge items, 6 each with 2-5 options. Items 1, 3
and 5 of each group are visible (synthesis and repair may see them); 2, 4 and 6
are held out and gate activation. Every item is asked in its original order and
in one rotation, which moves every option. Answers are judged by option text.

The items are a draft pending a native Arabic speaker's review; changing any
item means bumping SUITE_VERSION.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from decisioneval.models import Question

SUITE_VERSION = "canary-v1"
INSTRUCTION = "اختر الإجابة الصحيحة."
LETTERS = ("A", "B", "C", "D", "E")
NUMBERS = ("1", "2", "3", "4")
ORIGINAL = "original"
ROTATED = "rotated"


@dataclass(frozen=True)
class CanaryItem:
    id: str
    question: str
    options: tuple[str, ...]
    option_ids: tuple[str, ...]
    gold: str

    @property
    def gold_text(self) -> str:
        return self.options[self.option_ids.index(self.gold)]

    def as_question(self, order: str) -> Question:
        shown = self if order == ORIGINAL else rotated(self)
        return Question(
            id=f"{self.id}/{order}", dataset_name=SUITE_VERSION, category="canary", task="canary",
            instruction=INSTRUCTION, input=shown.question, options=shown.options,
            option_ids=shown.option_ids, gold_answer=shown.gold_text, gold_option_id=shown.gold,
        )


def _item(item_id: str, question: str, options: str, gold: str, ids: tuple[str, ...] = LETTERS) -> CanaryItem:
    texts = tuple(part.strip() for part in options.split("·"))
    return CanaryItem(item_id, question, texts, ids[: len(texts)], gold)


ITEMS: tuple[CanaryItem, ...] = (
    _item("2.1", "هل الرياض عاصمة المملكة العربية السعودية؟", "لا · نعم", "B"),
    _item("2.2", "هل التمر من الفواكه؟", "نعم · لا", "A"),
    _item("2.3", "هل يغلي الماء عند 100 درجة مئوية عند مستوى سطح البحر؟", "نعم · لا", "A"),
    _item("2.4", "هل تشرق الشمس من الغرب؟", "نعم · لا", "B"),
    _item("2.5", "هل القطة من الحيوانات الأليفة؟", "لا · نعم", "B"),
    _item("2.6", "هل عدد أيام الأسبوع سبعة؟", "نعم · لا", "A"),
    _item("3.1", "ما لون السماء في يوم صافٍ؟", "أزرق · أسود · أحمر", "A"),
    _item("3.2", "كم ثانية في الدقيقة الواحدة؟", "30 · 60 · 90", "B"),
    _item("3.3", "ما لون دم الإنسان السليم؟", "أصفر · أخضر · أحمر", "C"),
    _item("3.4", "كم عدد أرجل القطة؟", "أربع · اثنتان · ست", "A"),
    _item("3.5", "ما الحيوان الملقّب بسفينة الصحراء؟", "الحصان · الجمل · الفيل", "B"),
    _item("3.6", "ما ناتج 5 × 2؟", "7 · 8 · 10", "C"),
    _item("4.1", "ما ناتج جمع 15 + 27؟", "32 · 42 · 52 · 62", "2", NUMBERS),
    _item("4.2", "ما أكبر كوكب في المجموعة الشمسية؟", "المشتري · المريخ · عطارد · الزهرة", "A"),
    _item("4.3", "في أي قارة تقع مصر؟", "آسيا · أوروبا · أفريقيا · أستراليا", "C"),
    _item("4.4", "ما أقرب كوكب إلى الشمس؟", "المشتري · زحل · الأرض · عطارد", "D"),
    _item("4.5", "كم عدد أشهر السنة الميلادية؟", "10 · 11 · 12 · 13", "3", NUMBERS),
    _item("4.6", "ما الغاز الذي يحتاجه الإنسان للتنفس؟", "الهيليوم · الأكسجين · النيون · الأرغون", "B"),
    _item("5.1", "أي الكائنات التالية من الثدييات؟", "الصقر · التمساح · الدلفين · الأخطبوط · النعامة", "C"),
    _item("5.2", "ما عاصمة مصر؟", "القاهرة · الرباط · تونس · عمّان · الدوحة", "A"),
    _item("5.3", "كم عدد أيام الأسبوع؟", "5 · 7 · 6 · 8 · 9", "B"),
    _item("5.4", "ما اللون الناتج عن خلط الأزرق والأصفر؟", "الأحمر · البنفسجي · البرتقالي · الأخضر · الأسود", "D"),
    _item("5.5", "ما أطول نهر في أفريقيا؟", "الكونغو · النيجر · الزمبيزي · السنغال · النيل", "E"),
    _item("5.6", "كم عدد أضلاع المثلث؟", "ستة · ثلاثة · أربعة · خمسة · ثمانية", "B"),
)
VISIBLE = tuple(item for item in ITEMS if item.id.endswith((".1", ".3", ".5")))
HELD_OUT = tuple(item for item in ITEMS if item.id.endswith((".2", ".4", ".6")))
SEED_CHECK = tuple(item for item in ITEMS if item.id in ("2.1", "3.2", "4.3", "5.4"))
GATE = 9  # P(>= 9 of 12 held-out by chance) = 0.0021 < 0.01, while P(>= 8) = 0.0121


def rotated(item: CanaryItem) -> CanaryItem:
    """options[1:] + options[:1] with ids reassigned by position, so every option moves."""
    options = item.options[1:] + item.options[:1]
    return replace(item, options=options, gold=item.option_ids[options.index(item.gold_text)])


@dataclass(frozen=True)
class Answer:
    item: CanaryItem
    order: str
    choice: str | None  # an option id in the order shown, or None when invalid

    @property
    def shown(self) -> CanaryItem:
        return self.item if self.order == ORIGINAL else rotated(self.item)

    @property
    def chosen_text(self) -> str | None:
        shown = self.shown
        return shown.options[shown.option_ids.index(self.choice)] if self.choice in shown.option_ids else None


@dataclass(frozen=True)
class Verdict:
    verdict: str
    correct_original: int | None = None
    correct_rotated: int | None = None
    item_count: int = len(HELD_OUT)
    p_value: float | None = None
    structure_errors: tuple[str, ...] = ()
    behaviour_report: dict[str, Any] = field(default_factory=dict)

    def as_validation(self, split: str = "held_out") -> dict[str, Any]:
        return {
            "split": split,
            "suite_version": SUITE_VERSION,
            "verdict": self.verdict,
            "correct_original": self.correct_original,
            "correct_rotated": self.correct_rotated,
            "item_count": self.item_count,
            "p_value": self.p_value,
            "structure_errors": list(self.structure_errors),
            "behaviour_report": self.behaviour_report,
        }


def tail_probability(correct: int, items: tuple[CanaryItem, ...] = HELD_OUT) -> float:
    """P(at least `correct` right) for a picker choosing uniformly at random: a Poisson-binomial tail."""
    distribution = [1.0]
    for item in items:
        chance = 1.0 / len(item.options)
        nxt = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            nxt[count] += mass * (1 - chance)
            nxt[count + 1] += mass * chance
        distribution = nxt
    return sum(distribution[correct:])


def judge(answers: list[Answer], structure_errors: tuple[str, ...] = ()) -> Verdict:
    """Gate on the held-out items in both orders; report behaviour over every answer."""
    report = behaviour_report(answers)
    if structure_errors:
        return Verdict("structure_fail", structure_errors=structure_errors, behaviour_report=report)
    held = {item.id for item in HELD_OUT}

    def correct(order: str) -> int:
        return sum(
            1 for answer in answers
            if answer.item.id in held and answer.order == order and answer.chosen_text == answer.item.gold_text
        )

    original, turned = correct(ORIGINAL), correct(ROTATED)
    passed = original >= GATE and turned >= GATE
    return Verdict(
        "pass" if passed else "delivery_fail",
        correct_original=original,
        correct_rotated=turned,
        p_value=tail_probability(min(original, turned)),
        behaviour_report=report,
    )


def behaviour_report(answers: list[Answer]) -> dict[str, Any]:
    picks = Counter(
        str(answer.shown.option_ids.index(answer.choice) + 1)
        for answer in answers
        if answer.choice in answer.shown.option_ids
    )
    total = sum(picks.values())
    chosen = {(answer.item.id, answer.order): answer.chosen_text for answer in answers}
    disagreements = [
        {"item": item_id, "original": text, "rotated": chosen[(item_id, ROTATED)]}
        for (item_id, order), text in chosen.items()
        if order == ORIGINAL and (item_id, ROTATED) in chosen and chosen[(item_id, ROTATED)] != text
    ]
    misses = [
        {"item": answer.item.id, "expected": answer.item.gold_text, "chosen": answer.chosen_text, "order": answer.order}
        for answer in answers
        if answer.chosen_text != answer.item.gold_text
    ]
    # Held-out items and their answers stay inside the runner; the report only counts them.
    held = {item.id for item in HELD_OUT}
    return {
        "position_pick_rate": {position: count / total for position, count in sorted(picks.items())} if total else {},
        "rotation_disagreements": [row for row in disagreements if row["item"] not in held],
        "misses": [row for row in misses if row["item"] not in held],
        "held_out_rotation_disagreements": sum(row["item"] in held for row in disagreements),
        "held_out_misses": sum(row["item"] in held for row in misses),
    }
