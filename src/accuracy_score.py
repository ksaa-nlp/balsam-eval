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

# Define expanded choice normalization mappings with more Arabic variations
CHOICE_MAPPINGS = {
    # Arabic yes/no with expanded options
    "نعم": ["نعم", "أجل", "صحيح", "صح", "yes", "true", "أي نعم", "نعما", "بلى", "إيجابي", "موافق", "بالتأكيد", "حقا", "فعلا", "تماما"],
    "لا": ["لا", "كلا", "غير صحيح", "خطأ", "no", "false", "ليس", "لن", "سلبي", "غير موافق", "أبدا", "إطلاقا"],

    # Arabic true/false specific mapping
    "صح": ["صح", "صحيح", "true", "نعم", "correct", "حق", "حقيقي", "صادق"],
    "خطأ": ["خطأ", "غير صحيح", "false", "لا", "incorrect", "باطل", "كاذب", "غلط"],

    # English yes/no
    "yes": ["yes", "y", "true", "correct", "نعم", "أجل", "صح", "صحيح"],
    "no": ["no", "n", "false", "incorrect", "لا", "كلا", "خطأ", "غير صحيح"],

    # Multiple choice (expanded)
    "a": ["a", "أ", "ا", "option a", "الخيار أ", "اختيار أ", "a)", "(a)", "الإجابة أ", "الجواب أ"],
    "b": ["b", "ب", "option b", "الخيار ب", "اختيار ب", "b)", "(b)", "الإجابة ب", "الجواب ب"],
    "c": ["c", "ج", "option c", "الخيار ج", "اختيار ج", "c)", "(c)", "الإجابة ج", "الجواب ج"],
    "d": ["d", "د", "option d", "الخيار د", "اختيار د", "d)", "(d)", "الإجابة د", "الجواب د"]
}


def normalize_choice(text):
    """
    Enhanced choice normalization that aggressively extracts just the answer:
    - Boolean values
    - Yes/No questions in Arabic and English
    - True/False questions in Arabic and English
    - Multiple choice questions (A/B/C/D)
    - Arabic and English variations
    - Extracts answer choices from explanatory text
    """
    if isinstance(text, bool):
        return "نعم" if text else "لا"

    if not isinstance(text, str):
        text = str(text)

    # First, look for answers at the very beginning of the text (most common case)
    # Split by newlines and periods to get the first statement
    first_sentence = text.split('\n')[0].split('.')[0].strip()
    
    # Check if the first sentence is just an answer
    clean_first = first_sentence.lower().strip()
    clean_first = ''.join(ch for ch in clean_first if ch not in ALL_PUNCTUATIONS)
    
    # Direct match for simple answers at the beginning
    for standard_form, variations in CHOICE_MAPPINGS.items():
        if clean_first in variations:
            return standard_form
    
    # Check if first word/character is an answer
    first_word = clean_first.split()[0] if clean_first.split() else ""
    if first_word in CHOICE_MAPPINGS:
        return first_word
    
    # Check first character for multiple choice
    if len(clean_first) > 0:
        first_char = clean_first[0]
        if first_char in ['a', 'b', 'c', 'd', 'أ', 'ب', 'ج', 'د']:
            return first_char

    # Look for explicit answer patterns in the text in both Arabic and English
    choice_patterns = [
        # Arabic patterns - prioritize patterns at the beginning
        r'^([a-dأ-د])\b',  # Answer letter at the very start
        r'^(صح|صحيح|خطأ|غير صحيح)\b',  # True/false at start
        r'^(نعم|لا)\b',  # Yes/no at start
        
        # Then look for structured patterns
        r'\bالإجابة\s+(?:هي\s+)?[:\s]?\s*([a-dأ-د])\b',  # "الإجابة هي: أ"
        r'\bالخيار\s+(?:هو\s+)?[:\s]?\s*([a-dأ-د])\b',   # "الخيار هو: ب"
        r'\bالجواب\s+(?:هو\s+)?[:\s]?\s*([a-dأ-د])\b',   # "الجواب هو: ج"
        r'\bالإجابة\s+(?:هي\s+)?[:\s]?\s*(صح|صحيح|خطأ|غير صحيح)\b',
        r'\bالجواب\s+(?:هو\s+)?[:\s]?\s*(صح|صحيح|خطأ|غير صحيح)\b',
        r'\bالإجابة\s+(?:هي\s+)?[:\s]?\s*(نعم|لا)\b',
        r'\bالجواب\s+(?:هو\s+)?[:\s]?\s*(نعم|لا)\b',

        # English patterns
        r'^([a-d])\b',  # Answer letter at start
        r'^(true|false|yes|no)\b',  # Boolean at start
        r'\banswer\s+(?:is\s+)?[:\s]?\s*([a-dأ-د])\b',
        r'\boption\s+(?:is\s+)?[:\s]?\s*([a-dأ-د])\b',
        r'\bchoice\s+(?:is\s+)?[:\s]?\s*([a-dأ-د])\b',
        r'\banswer\s+(?:is\s+)?[:\s]?\s*(true|false|yes|no)\b',
        r'\banswer\s+(?:is\s+)?[:\s]?\s*(correct|incorrect)\b',
    ]

    # Try to extract choice from patterns
    for pattern in choice_patterns:
        match = re.search(pattern, text.lower())
        if match:
            extracted = match.group(1).lower()
            # Map variations to standardized forms
            if extracted in ["صح", "صحيح", "true", "correct"]:
                return "صح"
            elif extracted in ["خطأ", "غير صحيح", "false", "incorrect"]:
                return "خطأ"
            elif extracted in ["نعم", "yes"]:
                return "نعم"
            elif extracted in ["لا", "no"]:
                return "لا"
            else:
                return extracted

    # Clean text for further processing
    cleaned_text = text.lower().strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = ''.join(ch for ch in cleaned_text if ch not in ALL_PUNCTUATIONS)

    # Check for exact matches in the mapping
    for standard_form, variations in CHOICE_MAPPINGS.items():
        if cleaned_text in variations:
            return standard_form

    # Check for contained variations (words that might be part of a longer response)
    for standard_form, variations in CHOICE_MAPPINGS.items():
        for variation in variations:
            # Check if the variation is a standalone word at the beginning of text
            if re.search(r'\b' + re.escape(variation) + r'\b', cleaned_text):
                return standard_form

    # Look for first occurrence of a, b, c, d in the text
    match = re.search(r'\b([a-dأ-د])\b', cleaned_text)
    if match:
        return match.group(1).lower()

    # Special handling for Arabic yes/no/true/false patterns
    if any(word in cleaned_text for word in ["نعم", "أجل", "صحيح", "صح", "إيجابي", "موافق"]):
        return "نعم"
    if any(word in cleaned_text for word in ["لا", "كلا", "غير صحيح", "خطأ", "سلبي", "غير موافق"]):
        return "لا"

    # If all else fails, return the first word if it's short (likely an answer)
    first_word = cleaned_text.split()[0] if cleaned_text.split() else cleaned_text
    if len(first_word) <= 3:  # Short answers are more likely to be the actual answer
        return first_word

    return cleaned_text


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

        # Special handling for Arabic true/false and yes/no equivalence
        # Consider "صح" equivalent to "نعم" and "خطأ" equivalent to "لا" for scoring
        if (norm_ref in ["صح", "نعم"] and norm_pred in ["صح", "نعم"]) or \
           (norm_ref in ["خطأ", "لا"] and norm_pred in ["خطأ", "لا"]):
            is_correct = True
        else:
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
    output_type="generate_until",  # Changed from "multiple_choice" to "generate_until"
    aggregation="custom_accuracy",
)
def accuracy_fn(items):
    """Return items as is for the aggregation function."""
    return items


def process_results(doc, results):
    """
    Process results to extract predictions and references.
    Enhanced to handle Arabic yes/no, true/false and multiple choice responses.
    """
    print(f"Document: {doc}")
    print(f"Results: {results}")

    # Initialize default prediction
    pred = ""

    # Check if this is a multiple choice question, yes/no question, or true/false question
    instruction = doc.get("instruction", "")
    input_text = doc.get("input", "")
    full_context = f"{instruction} {input_text}"

    is_multiple_choice = any(opt in full_context.lower()
                             for opt in ["خيار", "option", "a)", "b)", "c)", "d)",
                                         "(a)", "(b)", "(c)", "(d)", "أ)", "ب)", "ج)", "د)",
                                         "(أ)", "(ب)", "(ج)", "(د)"])

    is_yes_no = any(opt in full_context.lower()
                    for opt in ["نعم", "لا", "yes", "no", "yes/no", "نعم أو لا", "نعم او لا"])

    is_true_false = any(opt in full_context.lower()
                        for opt in ["صح", "خطأ", "صحيح", "غير صحيح", "true", "false", "true/false", "صح أو خطأ", "صح او خطأ"])

    # For generate_until mode, results will be a string instead of logprobs/choices
    if isinstance(results, str):
        # Use the generated text as prediction
        pred = results.strip()
    else:
        # Fallback for other result formats (maintaining compatibility)
        if isinstance(results, list):
            if results:
                pred = str(results[0])
        elif isinstance(results, tuple) and len(results) == 2:
            pred = str(results[1])
        else:
            pred = str(results)

    # Get reference answer
    ref = doc.get("output", "")
    if isinstance(ref, list):
        ref = ref[0]

    # Normalize both reference and prediction
    norm_ref = normalize_choice(ref)
    norm_pred = normalize_choice(pred)

    # Determine question type for logging
    question_type = "Multiple Choice" if is_multiple_choice else \
        "Yes/No" if is_yes_no else \
        "True/False" if is_true_false else "Other"

    print(f"Processed - REF: {ref} ({norm_ref}) | PRED: {pred} ({norm_pred})")
    print(f"Question type: {question_type}")

    return {"accuracy": [(norm_ref, norm_pred)]}
