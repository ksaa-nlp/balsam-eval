import re

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore
from lm_eval.api.registry import register_aggregation, register_metric
from collections.abc import Iterable
import nltk
import evaluate
import pyarabic.araby as araby
import unicodedata
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from lm_eval.api.registry import register_aggregation, register_metric
import sacrebleu

# Download necessary resources
nltk.download('punkt_tab')

# Load Rouge evaluator from `evaluate` library
bleu = evaluate.load("bleu")
PUNCT_TABLE = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
ALL_PUNCTUATIONS += ''.join([o for o in others if o not in ALL_PUNCTUATIONS])

def prepare_texts(text, change_curly_braces, remove_diactrics):
    """
    Preprocess the text by handling punctuation, curly braces, and diacritics.
    """
    # 1. put spaces before and after each punctuation
    text = re.sub('([' + ALL_PUNCTUATIONS + '])', ' \\1 ', text)

    # 2. change all {} to []
    if change_curly_braces:
        text = text.replace('{', '[').replace('}', ']')

    # 3. Remove diacritics
    if remove_diactrics:
        text = araby.strip_diacritics(text)

    return text


def custem_bleu_aggregation(items):

    tokenizer = lambda x: x.split()
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    refs = [ref[0] if isinstance(ref, (list, tuple)) and len(ref) > 0 else ref for ref in refs]
    print(f"Refs before processing and if tuple corrected : {refs}")

    # Now apply text processing
    refs = [prepare_texts(ref, change_curly_braces=False, remove_diactrics=True).strip() for ref in refs]
    preds = [prepare_texts(pred, change_curly_braces=True, remove_diactrics=True).strip() for pred in preds]

    # Initialize sums for each BLEU score
    bleu_score= 0.0
   
    for i in range(len(refs)):
        score = bleu.compute(references=[refs[i]], predictions=[preds[i]], tokenizer=tokenizer)
    
        bleu_score += score["bleu"]


    count = len(refs)
    avg_bleu = bleu_score / count


    return avg_bleu


def process_results(doc, results):
    preds, golds = results[0], doc["output"]

    return {'bleu':[golds, preds]}
