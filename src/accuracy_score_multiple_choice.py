import re
import pyarabic.araby as araby
import unicodedata
import sys
from lm_eval.api.registry import register_aggregation, register_metric


# Punctuation table setup
PUNCT_TABLE = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ'''
ALL_PUNCTUATIONS += ''.join([o for o in others if o not in ALL_PUNCTUATIONS])

# Define choice normalization mappings
CHOICE_MAPPINGS = {
    # Arabic yes/no
    "نعم": ["نعم", "أجل", "صحيح", "yes", "true", "صح", "أي نعم", "نعما"],
    "لا": ["لا", "كلا", "غير صحيح", "no", "false", "خطأ", "ليس", "لن"],

    # English yes/no
    "yes": ["yes", "y", "true", "correct", "نعم", "أجل"],
    "no": ["no", "n", "false", "incorrect", "لا", "كلا"],

    # Multiple choice (expanded)
    "a": ["a", "أ", "ا", "option a", "الخيار أ", "اختيار أ", "a)", "(a)"],
    "b": ["b", "ب", "option b", "الخيار ب", "اختيار ب", "b)", "(b)"],
    "c": ["c", "ج", "option c", "الخيار ج", "اختيار ج", "c)", "(c)"],
    "d": ["d", "د", "option d", "الخيار د", "اختيار د", "d)", "(d)"]
}


def normalize_choice(text):
    """
    Enhanced choice normalization that handles:
    - Boolean values
    - Yes/No questions
    - Multiple choice questions (A/B/C/D)
    - Arabic and English variations
    """
    if isinstance(text, bool):
        return "نعم" if text else "لا"

    if not isinstance(text, str):
        text = str(text)

    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(ch for ch in text if ch not in ALL_PUNCTUATIONS)

    # Special handling for boolean-like strings
    if text in ['true', 'صحيح', 'صح', 'أجل', 'نعم', 'yes']:
        return "نعم"
    if text in ['false', 'غير صحيح', 'خطأ', 'لا', 'كلا', 'no']:
        return "لا"

    # Extract first character for multiple choice (handles "A)", "(A)", etc.)
    if len(text) > 0:
        first_char = text[0].lower()
        if first_char in ['a', 'b', 'c', 'd', 'أ', 'ب', 'ج', 'د']:
            return first_char

    for standard_form, variations in CHOICE_MAPPINGS.items():
        if text in variations:
            return standard_form

    return text


def prepare_texts(text, change_curly_braces=True, remove_diactrics=True):
    """
    Preprocess the text by handling punctuation, curly braces, and diacritics.
    """
    if not isinstance(text, str):
        text = str(text)

    text = re.sub('([' + re.escape(ALL_PUNCTUATIONS) + '])', r' \1 ', text)

    if change_curly_braces:
        text = text.replace('{', '[').replace('}', ']')

    if remove_diactrics:
        text = araby.strip_diacritics(text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


@register_aggregation("custom_accuracy")
def custom_accuracy_aggregation(items):
    """
    Calculate the accuracy score between predictions and references.
    """
    print(f"Items: {items}")
    if not items:
        return 0.0

    correct = 0
    total = len(items)

    for item in items:
        if isinstance(item, dict):
            ref = item.get("ref", "")
            pred = item.get("pred", "")
        elif isinstance(item, list) and len(item) == 1:
            ref, pred = item[0]
        elif isinstance(item, tuple) and len(item) == 2:
            ref, pred = item
        else:
            print(f"✗ Malformed item: {item}")
            continue

        norm_ref = normalize_choice(ref)
        norm_pred = normalize_choice(pred)

        is_correct = norm_ref == norm_pred

        if is_correct:
            correct += 1
            print(
                f"✓ CORRECT | REF: {ref} ({norm_ref}) | PRED: {pred} ({norm_pred})")
        else:
            print(
                f"✗ WRONG   | REF: {ref} ({norm_ref}) | PRED: {pred} ({norm_pred})")

    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return accuracy


@register_metric(
    metric="accuracy",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="custom_accuracy",
)
def accuracy_fn(items):
    """Return items as is for the aggregation function."""
    return items


def process_results(doc, results):
    """
    Process results to extract predictions and references.
    Handles both yes/no questions and multiple choice questions.
    """
    print(f"Document: {doc}")
    print(f"Results: {results}")

    # Initialize default prediction
    pred = ""

    # Check if this is a multiple choice question (options A/B/C/D in instruction)
    is_multiple_choice = any(opt in doc.get("instruction", "").lower()
                             for opt in ["خيار", "option", "a", "b", "c", "d"])

    # Handle different result formats
    if isinstance(results, list):
        if all(isinstance(r, tuple) and len(r) == 2 for r in results):
            # For multiple choice, select the option with highest probability
            if is_multiple_choice:
                try:
                    # Get index of option with highest logprob
                    best_idx = max(enumerate(results),
                                   key=lambda x: x[1][0])[0]
                    pred = ["a", "b", "c", "d"][best_idx]
                except:
                    pred = "a"  # Default to first option if error occurs
            else:
                # For yes/no, take the boolean with highest probability
                best_pred = max(results, key=lambda x: x[0])
                if isinstance(best_pred[1], bool):
                    pred = "نعم" if best_pred[1] else "لا"
                else:
                    pred = str(best_pred[1])
        else:
            # Fallback to first result if format is unexpected
            pred = results[0] if isinstance(
                results[0], str) else str(results[0])
    elif isinstance(results, tuple) and len(results) == 2:
        if is_multiple_choice:
            # For multiple choice with single tuple, assume it's for option A
            pred = "a"
        else:
            # For yes/no with single tuple
            if isinstance(results[1], bool):
                pred = "نعم" if results[1] else "لا"
            else:
                pred = str(results[1])
    else:
        # Fallback for other cases
        pred = str(results)

    # Get reference answer
    ref = doc.get("output", "")
    if isinstance(ref, list):
        ref = ref[0]

    # Normalize both reference and prediction
    norm_ref = normalize_choice(ref)
    norm_pred = normalize_choice(pred)

    print(f"Processed - REF: {ref} ({norm_ref}) | PRED: {pred} ({norm_pred})")
    print(
        f"Question type: {'Multiple Choice' if is_multiple_choice else 'Yes/No'}")

    return {"accuracy": [(norm_ref, norm_pred)]}
