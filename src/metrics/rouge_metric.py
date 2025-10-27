"""ROUGE metric implementation - all ROUGE functionality in one place."""

import re
import unicodedata
import sys
import evaluate
import pyarabic.araby as araby
from lm_eval.api.registry import register_aggregation, register_metric

# Load Rouge evaluator
rouge = evaluate.load("rouge")

# Punctuation table setup
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in others if o not in ALL_PUNCTUATIONS])


def prepare_texts(text, change_curly_braces=False, remove_diacritics=True):
    """Preprocess text for ROUGE evaluation."""
    # Put spaces before and after each punctuation
    text = re.sub("([" + ALL_PUNCTUATIONS + "])", " \\1 ", text)
    
    # Change all {} to []
    if change_curly_braces:
        text = text.replace("{", "[").replace("}", "]")
    
    # Remove diacritics
    if remove_diacritics:
        text = araby.strip_diacritics(text)
    
    return text


@register_aggregation("rouge")
def rouge_aggregation(items):
    """Aggregate ROUGE scores across the dataset."""
    def tokenizer(x): return x.split()
    
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    
    # Preprocess the texts
    refs = [
        prepare_texts(ref, change_curly_braces=False, remove_diacritics=True).strip()
        for ref in refs
    ]
    preds = [
        prepare_texts(pred, change_curly_braces=True, remove_diacritics=True).strip()
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
    """Return items as is for the aggregation function."""
    return items


def process_results(doc, results):
    """Process results to extract predictions and references."""
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]
    return {"rouge": [(golds, preds)]}
