"""BLEU metric implementation."""

import re
import evaluate
import pyarabic.araby as araby
import unicodedata
import sys
from lm_eval.api.registry import register_aggregation, register_metric

# Load BLEU evaluator
bleu = evaluate.load("bleu")

# Punctuation table
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ'''
ALL_PUNCTUATIONS += ''.join([o for o in others if o not in ALL_PUNCTUATIONS])


def prepare_texts(text, change_curly_braces=False, remove_diacritics=True):
    """Preprocess text for BLEU evaluation."""
    # Put spaces before and after each punctuation
    text = re.sub('([' + ALL_PUNCTUATIONS + '])', ' \\1 ', text)
    
    # Change all {} to []
    if change_curly_braces:
        text = text.replace('{', '[').replace('}', ']')
    
    # Remove diacritics
    if remove_diacritics:
        text = araby.strip_diacritics(text)
    
    return text


@register_aggregation("custom_bleu")
def bleu_aggregation(items):
    """Aggregate BLEU scores."""
    def tokenizer(x): return x.split()
    
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    
    # Handle tuple references
    refs = [ref[0] if isinstance(ref, (list, tuple)) and len(ref) > 0 else ref for ref in refs]
    
    # Preprocess texts
    refs = [
        prepare_texts(ref, change_curly_braces=False, remove_diacritics=True).strip()
        for ref in refs
    ]
    preds = [
        prepare_texts(pred, change_curly_braces=True, remove_diacritics=True).strip()
        for pred in preds
    ]
    
    bleu_score = 0.0
    valid_count = 0
    
    for i in range(len(refs)):
        # Skip empty predictions or references
        if len(preds[i]) == 0 or len(refs[i]) == 0:
            continue
        
        try:
            score = bleu.compute(
                references=[refs[i]], 
                predictions=[preds[i]], 
                tokenizer=tokenizer
            )
            bleu_score += score["bleu"]
            valid_count += 1
        except (ZeroDivisionError, Exception):
            continue
    
    count = valid_count if valid_count > 0 else 1
    avg_bleu = bleu_score / count
    
    return avg_bleu 


# @register_metric(
#     metric="bleu",
#     higher_is_better=True,
#     output_type="generate_until",
#     aggregation="bleu",
# )
# def bleu_fn(items):
#     """Return items as is for the aggregation function."""
#     return items


def process_results(doc, results):
    """Process results for BLEU."""
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]
    return {'bleu': [(golds, preds)]}
