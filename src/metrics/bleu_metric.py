import re

import evaluate
import pyarabic.araby as araby
import unicodedata
import sys
from lm_eval.api.registry import register_aggregation

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

@register_aggregation("custom_bleu")
def bleu_aggregation(items):
    def tokenizer(x): return x.split()
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    refs = [ref[0] if isinstance(ref, (list, tuple)) and len(
        ref) > 0 else ref for ref in refs]
    print(f"Refs before processing and if tuple corrected : {refs}")

    # Now apply text processing
    refs = [prepare_texts(ref, change_curly_braces=False,
                          remove_diactrics=True).strip() for ref in refs]
    preds = [prepare_texts(pred, change_curly_braces=True,
                           remove_diactrics=True).strip() for pred in preds]

    # Debug: Check for empty predictions after processing
    empty_preds = [(i, pred) for i, pred in enumerate(preds) if len(pred) == 0]
    if empty_preds:
        print(f"WARNING: Found {len(empty_preds)} empty predictions after processing:")
        for idx, pred in empty_preds[:5]:  # Show first 5 empty predictions
            print(f"  Index {idx}: '{pred}' (original ref: '{refs[idx]}')")

    # Initialize sums for each BLEU score
    bleu_score = 0.0
    valid_count = 0

    for i in range(len(refs)):
        # Check if prediction is empty after processing
        if len(preds[i]) == 0:
            print(f"Skipping empty prediction at index {i}")
            # Option 1: Skip empty predictions (recommended)
            continue
            
            # Option 2: Assign score of 0.0 for empty predictions
            # bleu_score += 0.0
            # valid_count += 1
            # continue

        # Check if reference is empty (shouldn't happen but good to check)
        if len(refs[i]) == 0:
            print(f"Skipping empty reference at index {i}")
            continue

        try:
            score = bleu.compute(references=[refs[i]], predictions=[
                                 preds[i]], tokenizer=tokenizer)
            bleu_score += score["bleu"]
            valid_count += 1
        except ZeroDivisionError as e:
            print(f"BLEU computation failed at index {i}: {e}")
            print(f"  Prediction: '{preds[i]}'")
            print(f"  Reference: '{refs[i]}'")
            # Skip this sample or assign 0.0
            continue

    # Use valid_count instead of total count
    count = valid_count if valid_count > 0 else 1  # Avoid division by zero
    avg_bleu = bleu_score / count
    
    print(f"Computed BLEU on {valid_count}/{len(refs)} valid samples")
    
    return avg_bleu


def process_results(doc, results):
    preds, golds = results[0], doc["output"]

    return {'bleu': [golds, preds]}