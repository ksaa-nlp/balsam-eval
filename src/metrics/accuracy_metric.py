"""Accuracy metric with enhanced answer extraction."""

import re
import unicodedata
import sys
from lm_eval.api.registry import register_aggregation, register_metric

# Punctuation table setup
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ'''
ALL_PUNCTUATIONS += ''.join([o for o in others if o not in ALL_PUNCTUATIONS])


def extract_first_word_or_line(text):
    """Extract the first meaningful content from text."""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip()
    if not text:
        return ""
    
    # Get first line
    first_line = text.split('\n')[0].strip()
    first_line = re.sub(r'[.،؟!]+$', '', first_line)
    
    # If short, return it
    if len(first_line.split()) <= 3:
        return first_line
    
    # Otherwise, get first word
    first_word = first_line.split()[0] if first_line.split() else first_line
    first_word = re.sub(r'^["\'""]|["\'""]$', '', first_word)
    first_word = re.sub(r'[.،؟!]+$', '', first_word)
    
    return first_word


def extract_options_from_question(question_text):
    """Extract possible answer options from question text."""
    if not question_text:
        return []
    
    options = set()
    
    # Check for yes/no questions
    if 'يجب أن يكون الجواب بـ"نعم" أو "لا"' in question_text:
        return ['نعم', 'لا']
    
    # Multiple choice pattern (A), (B), (C), (D)
    mc_matches = re.findall(r'\([A-Za-z]\)', question_text)
    if mc_matches:
        for match in mc_matches:
            letter = match.strip('()')
            options.add(letter)
            options.add(match)
    
    # "أم" or "or" patterns
    or_patterns = [
        r'(\w+)\s+(?:أم|أو)\s+(\w+)',
        r'(\w+)\s+or\s+(\w+)',
        r'"([^"]+)"\s+(?:أو|أم|or)\s+"([^"]+)"',
    ]
    
    for pattern in or_patterns:
        matches = re.findall(pattern, question_text)
        for match in matches:
            options.update([word.strip().strip('"') for word in match if word.strip()])
    
    # Quoted options
    quoted_matches = re.findall(r'["\'""]([^"\'""]+)["\'""]', question_text)
    options.update([opt.strip() for opt in quoted_matches if opt.strip()])
    
    # Binary pairs
    binary_pairs = [
        ['نعم', 'لا'], ['yes', 'no'],
        ['صح', 'خطأ'], ['true', 'false'],
    ]
    
    text_lower = question_text.lower()
    for pair in binary_pairs:
        if any(word in text_lower for word in pair):
            options.update(pair)
            break
    
    # Clean options
    clean_options = []
    for opt in options:
        cleaned = opt.strip().strip('"\'"".،,')
        if cleaned and len(cleaned) > 0:
            clean_options.append(cleaned)
    
    return clean_options


def simple_normalize(text, reference_options=None):
    """Normalize text for comparison."""
    if isinstance(text, bool):
        return "نعم" if text else "لا"
    
    if not isinstance(text, str):
        text = str(text)
    
    extracted = extract_first_word_or_line(text)
    clean_extracted = re.sub(r'[()[\]{}]', '', extracted).strip()
    
    # Check for multiple choice answer (A), B), etc.
    mc_match = re.match(r'^([A-Za-z])\)', extracted.strip())
    if mc_match:
        letter_only = mc_match.group(1)
        if reference_options:
            for option in reference_options:
                if option.strip().upper() == letter_only.upper():
                    return option
        return letter_only
    
    # Check if just a letter
    if len(extracted.strip()) == 1 and extracted.strip().isalpha():
        return extracted.strip()
    
    # Try to match with reference options
    if reference_options:
        extracted_lower = extracted.lower().strip()
        clean_extracted_lower = clean_extracted.lower().strip()
        
        for option in reference_options:
            option_lower = option.lower().strip()
            clean_option = re.sub(r'[()[\]{}]', '', option_lower).strip()
            
            if extracted_lower == option_lower or clean_extracted_lower == clean_option:
                return option
            
            if clean_option in clean_extracted_lower or clean_extracted_lower in clean_option:
                return option
    
    return clean_extracted if clean_extracted else extracted


@register_aggregation("custom_accuracy")
def custom_accuracy_aggregation(items):
    """Calculate accuracy score."""
    if not items:
        return 0.0
    
    correct = 0
    total = len(items)
    
    for item in items:
        # Handle different item formats
        if isinstance(item, dict):
            ref = item.get("ref", "")
            pred = item.get("pred", "")
            context = item.get("context", "")
        elif isinstance(item, (list, tuple)):
            if len(item) == 1 and isinstance(item[0], (list, tuple)) and len(item[0]) >= 2:
                inner_item = item[0]
                ref = inner_item[0]
                pred = inner_item[1]
                context = inner_item[2] if len(inner_item) > 2 else ""
            elif len(item) >= 2:
                ref = item[0]
                pred = item[1]
                context = item[2] if len(item) > 2 else ""
            else:
                continue
        else:
            continue
        
        # Extract options and normalize
        context_options = extract_options_from_question(str(context))
        norm_ref = simple_normalize(ref, context_options)
        norm_pred = simple_normalize(pred, context_options)
        
        # Compare
        clean_ref = re.sub(r'[()[\]{}]', '', norm_ref).lower().strip()
        clean_pred = re.sub(r'[()[\]{}]', '', norm_pred).lower().strip()
        
        if clean_ref == clean_pred:
            correct += 1
    
    accuracy = (correct / total) if total > 0 else 0.0
    return accuracy


@register_metric(
    metric="accuracy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="custom_accuracy",
)
def accuracy_fn(items):
    """Return items as is for the aggregation function."""
    return items


def process_results(doc, results):
    """Process results for accuracy."""
    instruction = doc.get("instruction", "")
    input_text = doc.get("input", "")
    full_context = f"{instruction} {input_text}".strip()
    
    pred = ""
    if isinstance(results, str):
        pred = results.strip()
    elif isinstance(results, list) and results:
        pred = str(results[0]).strip()
    elif isinstance(results, dict):
        pred = str(results.get('text', results.get('generated_text', results.get('prediction', '')))).strip()
    else:
        pred = str(results).strip()
    
    ref = doc.get("output", "")
    if isinstance(ref, list) and ref:
        ref = str(ref[0])
    else:
        ref = str(ref)
    
    return {"accuracy": [(ref, pred, full_context)]}