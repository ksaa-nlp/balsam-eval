"""Post-processing utilities for metrics."""

import re

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore
from lm_eval.api.registry import register_aggregation, register_metric

from . import rouge_scorer

import evaluate
import pyarabic.araby as araby
import unicodedata
import sys

# Load Rouge evaluator from `evaluate` library
rouge = evaluate.load("rouge")
# Punctuation table
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in others if o not in ALL_PUNCTUATIONS])


def prepare_texts(text, change_curly_braces, remove_diactrics):
    """
    Preprocess the text by handling punctuation, curly braces, and diacritics.
    """
    # 1. put spaces before and after each punctuation
    text = re.sub("([" + ALL_PUNCTUATIONS + "])", " \\1 ", text)

    # 2. change all {} to []
    if change_curly_braces:
        text = text.replace("{", "[").replace("}", "]")

    # 3. Remove diacritics
    if remove_diactrics:
        text = araby.strip_diacritics(text)

    return text


def get_answers(doc):
    """
    Extract answers from the document, ensuring no duplicates.
    """
    answers = []
    answers_set = set()
    candidates = [doc["output"]]
    for candidate in candidates:
        answer = candidate
        if answer in answers_set:
            continue
        answers_set.add(answer)
        answers.append(answer)
    return answers


@register_aggregation("rouge")
def rouge_aggregation(items):
    """
    Aggregate the Rouge scores across the dataset for all types of ROUGE.
    """
    # print("self.name",self.name)
    def tokenizer(x): return x.split()
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    # Preprocess the texts
    refs = [
        prepare_texts(ref, change_curly_braces=False,
                      remove_diactrics=True).strip()
        for ref in refs
    ]
    preds = [
        prepare_texts(pred, change_curly_braces=True,
                      remove_diactrics=True).strip()
        for pred in preds
    ]

    # Initialize sums for each ROUGE score
    rouge1 = 0.0
    rouge2 = 0.0
    rougeL = 0.0
    rougeLsum = 0.0

    for i in range(len(refs)):
        # Compute ROUGE scores
        score = rouge.compute(
            references=[refs[i]], predictions=[preds[i]], tokenizer=tokenizer
        )

        # # Print each score for debugging
        # print(f"Reference: {refs[i]}")
        # print(f"Prediction: {preds[i]}")
        # print(f"ROUGE Scores: {score}")

        # Aggregate each type of ROUGE score
        rouge1 += score["rouge1"]
        rouge2 += score["rouge2"]
        rougeL += score["rougeL"]
        rougeLsum += score["rougeLsum"]

    count = len(refs)
    avg_rouge1 = rouge1 / count
    avg_rouge2 = rouge2 / count
    avg_rougeL = rougeL / count
    avg_rougeLsum = rougeLsum / count

    return {
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "rougeLsum": avg_rougeLsum,
    }


@register_metric(
    metric="rouge",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="rouge",
)
def rouge_fn(items):
    return items


def process_docs(dataset):
    def _process(doc):
        if "instruction" in doc:
            return {
                "id": doc["id"],
                "input": doc["instruction"] + "\n" + doc["input"],
                "answers": get_answers(doc),
            }

        return {
            "id": doc["id"],
            "input": doc["input"],
            "answers": get_answers(doc),
        }

    return dataset.map(_process)


def process_results(doc, results):
    preds, golds = results, doc["answers"]
    max_em = 0
    max_f1 = 0
    for gold_answer in golds:
        exact_match, f1_score = get_f1(preds, gold_answer), get_accuracy(
            preds, gold_answer
        )
        if gold_answer[0].strip():
            max_em = max(max_em, exact_match)
            max_f1 = max(max_f1, f1_score)
    return {"exact_match": max_em, "f1": max_f1, "accuracy": max_em}


def get_f1(predicted, gold):
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns F1 metric for the prediction.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return f1


def get_accuracy(predicted, gold):
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match for the prediction.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    em_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    em = np.mean(em_per_bag)
    em = round(em, 2)
    return em


def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted, gold):
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_answers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(
                    pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_answers_if_present(gold_bag, predicted_bag):
    gold_answer = set()
    predicted_answer = set()
    for gold_token in gold_bag:
        gold_answer.add(gold_token)
    for predicted_token in predicted_bag:
        predicted_answer.add(predicted_token)

    if (not gold_answer) or gold_answer.intersection(predicted_answer):
        return True
    return False


def _white_space_fix(text):
    return " ".join(text.split())


def _tokenize(text):
    return re.split(" ", text)


def _normalize(answer):
    tokens = [_white_space_fix(token.lower()) for token in _tokenize(answer)]
    tokens = [token for token in tokens if token.strip()]
    normalized = " ".join(tokens).strip()
    return normalized
