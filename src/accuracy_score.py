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
    first_line = text.split('\n')[0].strip()
    
    # Remove common punctuation from the end
    first_line = re.sub(r'[.،؟!]+$', '', first_line)
    
    # If the first line is short (likely just an answer), return it
    if len(first_line.split()) <= 3:
        return first_line
    
    # Otherwise, try to get the first word
    first_word = first_line.split()[0] if first_line.split() else first_line
    
    # Clean the first word
    first_word = re.sub(r'^["\'""]|["\'""]$', '', first_word)  # Remove quotes
    first_word = re.sub(r'[.،؟!]+$', '', first_word)  # Remove ending punctuation
    
    return first_word

def extract_options_from_question(question_text):
    """
    Extract possible answer options from the question text.
    """
    if not question_text:
        return []
    
    options = set()
    text_lower = question_text.lower()
    
    # Look for explicit yes/no instructions in quotes
    if 'يجب أن يكون الجواب بـ"نعم" أو "لا"' in question_text or 'يجب أن يكون الجواب بـ"نعم" أو "لا"' in question_text:
        return ['نعم', 'لا']
    
    # Pattern 1: "أم" or "or" patterns
    # Looking for "X أم Y" or "X or Y"
    or_patterns = [
        r'(\w+)\s+(?:أم|أو)\s+(\w+)',
        r'(\w+)\s+or\s+(\w+)',
        r'"([^"]+)"\s+(?:أو|أم|or)\s+"([^"]+)"',  # Quoted options
    ]
    
    for pattern in or_patterns:
        matches = re.findall(pattern, question_text)  # Use original text for quotes
        for match in matches:
            options.update([word.strip().strip('"') for word in match if word.strip()])
    
    # Pattern 2: Quoted options
    quoted_matches = re.findall(r'["\'""]([^"\'""]+)["\'""]', question_text)
    options.update([opt.strip() for opt in quoted_matches if opt.strip()])
    
    # Pattern 3: After "إذا كانت" or similar
    condition_patterns = [
        r'إذا كان[ت]?\s+([^.؟!]+)',
        r'whether (?:it )?(?:is |was )?([^.?!]+)',
    ]
    
    for pattern in condition_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Split by "أم" or "or"
            parts = re.split(r'\s+(?:أم|أو|or)\s+', match)
            options.update([part.strip() for part in parts if part.strip()])
    
    # Common binary pairs - if we detect the context, add both options
    binary_pairs = [
        ['نعم', 'لا'], ['yes', 'no'],
        ['صح', 'خطأ'], ['true', 'false'],
        ['عادية', 'مزعجة'], ['normal', 'annoying'],
        ['جيد', 'سيء'], ['good', 'bad'],
        ['إيجابي', 'سلبي'], ['positive', 'negative'],
    ]
    
    for pair in binary_pairs:
        if any(word in text_lower for word in pair):
            options.update(pair)
            break
    
    # Convert to list and clean
    clean_options = []
    for opt in options:
        cleaned = opt.strip().strip('"\'"".،,')
        if cleaned and len(cleaned) > 0:
            clean_options.append(cleaned)
    
    return clean_options

def simple_normalize(text, reference_options=None):
    """
    Simple normalization that tries to extract the core answer.
    """
    if isinstance(text, bool):
        return "نعم" if text else "لا"
    
    if not isinstance(text, str):
        text = str(text)
    
    # Get the first meaningful part
    extracted = extract_first_word_or_line(text)
    
    # If we have reference options, try to match
    if reference_options:
        extracted_lower = extracted.lower().strip()
        
        # Direct match
        for option in reference_options:
            if extracted_lower == option.lower().strip():
                return option
        
        # Partial match
        for option in reference_options:
            option_lower = option.lower().strip()
            if (option_lower in extracted_lower or 
                extracted_lower in option_lower):
                return option
    
    return extracted

@register_aggregation("custom_accuracy")
def custom_accuracy_aggregation(items):
    """
    Calculate the accuracy score between predictions and references.
    """
    print(f"\n=== ACCURACY AGGREGATION DEBUG ===")
    print(f"Total items received: {len(items)}")
    
    if not items:
        return 0.0

    correct = 0
    total = len(items)

    for i, item in enumerate(items):
        print(f"\n--- Item {i+1} ---")
        print(f"Raw item: {item}")
        print(f"Item type: {type(item)}")
        
        # Handle different item formats
        if isinstance(item, dict):
            ref = item.get("ref", "")
            pred = item.get("pred", "")
            context = item.get("context", "")
        elif isinstance(item, (list, tuple)):
            if len(item) == 1 and isinstance(item[0], (list, tuple)) and len(item[0]) >= 2:
                # Handle nested structure like [('ref', 'pred', 'context')]
                inner_item = item[0]
                ref = inner_item[0]
                pred = inner_item[1]
                context = inner_item[2] if len(inner_item) > 2 else ""
            elif len(item) >= 2:
                # Handle direct structure like ('ref', 'pred', 'context')
                ref = item[0]
                pred = item[1]
                context = item[2] if len(item) > 2 else ""
            else:
                print(f"✗ SKIPPED - List/tuple too short: {len(item)} items")
                continue
        else:
            print(f"✗ SKIPPED - Malformed item format: {type(item)}")
            continue

        print(f"Reference: '{ref}'")
        print(f"Prediction: '{pred}'")
        print(f"Context: '{context[:100]}...' " if len(str(context)) > 100 else f"Context: '{context}'")
        
        # Extract options from context
        context_options = extract_options_from_question(str(context))
        print(f"Extracted options: {context_options}")
        
        # Normalize
        norm_ref = simple_normalize(ref, context_options)
        norm_pred = simple_normalize(pred, context_options)
        
        print(f"Normalized Reference: '{norm_ref}'")
        print(f"Normalized Prediction: '{norm_pred}'")
        
        # Check match (case insensitive)
        is_correct = norm_ref.lower().strip() == norm_pred.lower().strip()
        
        if is_correct:
            correct += 1
            print(f"✓ CORRECT")
        else:
            print(f"✗ WRONG")

    accuracy = (correct / total) if total > 0 else 0.0
    print(f"\n=== FINAL RESULT ===")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy

@register_metric(
    metric="accuracy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="custom_accuracy",
)
def accuracy_fn(items):
    """Return items as is for the aggregation function."""
    print(f"accuracy_fn called with {len(items)} items")
    return items

def process_results(doc, results):
    """
    Process results to extract predictions and references.
    """
    print(f"\n=== PROCESS_RESULTS DEBUG ===")
    print(f"Document keys: {list(doc.keys()) if isinstance(doc, dict) else 'Not a dict'}")
    print(f"Document: {doc}")
    print(f"Results type: {type(results)}")
    print(f"Results: {results}")

    # Get the full context
    instruction = doc.get("instruction", "")
    input_text = doc.get("input", "")
    full_context = f"{instruction} {input_text}".strip()
    
    print(f"Full context: '{full_context}'")

    # Extract prediction from results
    pred = ""
    if isinstance(results, str):
        pred = results.strip()
    elif isinstance(results, list) and results:
        pred = str(results[0]).strip()
    elif isinstance(results, dict):
        # Handle different result dictionary formats
        pred = str(results.get('text', results.get('generated_text', results.get('prediction', '')))).strip()
    else:
        pred = str(results).strip()

    # Get reference answer
    ref = doc.get("output", "")
    if isinstance(ref, list) and ref:
        ref = str(ref[0])
    else:
        ref = str(ref)

    print(f"Extracted prediction: '{pred}'")
    print(f"Reference answer: '{ref}'")
    
    # Return the data for aggregation
    result_data = {
        "accuracy": [(ref, pred, full_context)]
    }
    
    print(f"Returning: {result_data}")
    return result_data