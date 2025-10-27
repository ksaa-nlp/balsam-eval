import re
import unicodedata
import sys
from lm_eval.api.registry import register_aggregation, register_metric

# Punctuation table setup
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in others if o not in ALL_PUNCTUATIONS])


def extract_first_word_or_line(text):
    """
    Simple extraction of the first meaningful content from text.
    This is our fallback to get just the answer.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return ""

    # Try to get the first line
    first_line = text.split("\n")[0].strip()

    # Remove common punctuation from the end
    first_line = re.sub(r"[.،؟!]+$", "", first_line)

    # If the first line is short (likely just an answer), return it
    if len(first_line.split()) <= 3:
        return first_line

    # Otherwise, try to get the first word
    first_word = first_line.split()[0] if first_line.split() else first_line

    # Clean the first word
    first_word = re.sub(r'^["\'""]|["\'""]$', "", first_word)  # Remove quotes
    first_word = re.sub(r"[.،؟!]+$", "", first_word)  # Remove ending punctuation

    return first_word

def simple_normalize(text, reference_options=None):
    """
    Simple normalization that tries to extract the core answer.
    Enhanced to handle parentheses and multiple choice answers.
    """
    if isinstance(text, bool):
        return "نعم" if text else "لا"

    if not isinstance(text, str):
        text = str(text)

    # Get the first meaningful part
    extracted = extract_first_word_or_line(text)

    # Clean extracted text - remove parentheses and common punctuation
    clean_extracted = re.sub(r"[()[\]{}]", "", extracted).strip()

    # NEW: Check if the extracted text starts with a letter followed by ) - this is likely a multiple choice answer
    mc_pattern = r"^([A-Za-z])\)"
    mc_match = re.match(mc_pattern, extracted.strip())
    if mc_match:
        # Extract just the letter part
        letter_only = mc_match.group(1)
        # If reference options include this letter, return it
        if reference_options:
            for option in reference_options:
                if option.strip().upper() == letter_only.upper():
                    return option
        return letter_only

    # NEW: Check if the extracted text is just a letter (A, B, C, D, etc.)
    if len(extracted.strip()) == 1 and extracted.strip().isalpha():
        return extracted.strip()

    # If we have reference options, try to match
    if reference_options:
        extracted_lower = extracted.lower().strip()
        clean_extracted_lower = clean_extracted.lower().strip()

        # Direct match with original extracted text
        for option in reference_options:
            option_lower = option.lower().strip()
            if extracted_lower == option_lower:
                return option

        # Direct match with cleaned extracted text (without parentheses)
        for option in reference_options:
            option_lower = option.lower().strip()
            clean_option = re.sub(r"[()[\]{}]", "", option_lower).strip()
            if clean_extracted_lower == clean_option:
                return option

        # NEW: Check if any reference option is a single letter that matches the start of our extracted text
        for option in reference_options:
            option_stripped = option.strip()
            if (
                len(option_stripped) == 1
                and option_stripped.isalpha()
                and extracted.strip().upper().startswith(option_stripped.upper())
            ):
                return option

        # Partial match - check if cleaned versions contain each other
        for option in reference_options:
            option_lower = option.lower().strip()
            clean_option = re.sub(r"[()[\]{}]", "", option_lower).strip()

            # Check both directions for containment
            if (
                clean_option in clean_extracted_lower
                or clean_extracted_lower in clean_option
            ):
                return option

    return clean_extracted if clean_extracted else extracted


@register_aggregation("accuracy")
def accuracy_aggregation(items):
    """
    Calculate the accuracy score between predictions and references.
    """

    if not items:
        return 0.0

    correct = 0
    total = len(items)

    for item in items:
        # Handle different item formats
        if isinstance(item, dict):
            ref = item.get("ref", "")
            pred = item.get("pred", "")
        elif isinstance(item, (list, tuple)):
            if (
                len(item) == 1
                and isinstance(item[0], (list, tuple))
            ):
                inner_item = item[0]
                if len(inner_item) >= 2:
                    ref = inner_item[0]
                    pred = inner_item[1]
                else:
                    continue
            elif len(item) >= 2:
                ref = item[0]
                pred = item[1]
            else:
                continue
        else:
            continue

        # Normalize WITHOUT context
        norm_ref = simple_normalize(ref)
        norm_pred = simple_normalize(pred)

        # Check match (case insensitive, and handle parentheses)
        clean_ref = re.sub(r"[()[\]{}]", "", norm_ref).lower().strip()
        clean_pred = re.sub(r"[()[\]{}]", "", norm_pred).lower().strip()

        is_correct = clean_ref == clean_pred

        if is_correct:
            correct += 1

    accuracy = (correct / total) if total > 0 else 0.0

    return accuracy


@register_metric(
    metric="accuracy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="accuracy",
)
def accuracy_fn(items):
    """Return items as is for the aggregation function."""
    return items


def process_results(doc, results):
    """
    Process results to extract predictions and references.
    """

    # Extract prediction from results
    pred = ""
    if isinstance(results, str):
        pred = results.strip()
    elif isinstance(results, list) and results:
        pred = str(results[0]).strip()
    elif isinstance(results, dict):
        # Handle different result dictionary formats
        pred = str(
            results.get(
                "text", results.get("generated_text", results.get("prediction", ""))
            )
        ).strip()
    else:
        pred = str(results).strip()

    # Get reference answer
    ref = doc.get("output", "")
    if isinstance(ref, list) and ref:
        ref = str(ref[0])
    else:
        ref = str(ref)

    # Return the data for aggregation
    # UPDATED: Format is now [ref, pred]
    result_data = {"accuracy": [ref, pred]}

    return result_data