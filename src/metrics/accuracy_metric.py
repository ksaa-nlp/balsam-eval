import re
import unicodedata
import sys
import logging
from lm_eval.api.registry import register_aggregation, register_metric
from lm_eval.api import registry as le_registry
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

# ---------- logging setup ----------
logger = logging.getLogger(__name__)

# ---------- punctuation setup ----------
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in others if o not in ALL_PUNCTUATIONS])


# ---------- helpers ----------
def extract_first_word_or_line(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        logger.debug("extract_first_word_or_line: Empty text after strip")
        return ""
    
    first_line = text.split("\n")[0].strip()
    first_line = re.sub(r"[.،؟!]+$", "", first_line)
    
    logger.debug(f"extract_first_word_or_line: first_line='{first_line}'")
    
    if len(first_line.split()) <= 3:
        logger.debug(f"extract_first_word_or_line: Returning short line (≤3 words): '{first_line}'")
        return first_line
    
    first_word = first_line.split()[0] if first_line.split() else first_line
    first_word = re.sub(r'^["\'""]|["\'""]$', "", first_word)
    first_word = re.sub(r"[.،؟!]+$", "", first_word)
    
    logger.debug(f"extract_first_word_or_line: Extracted first word: '{first_word}'")
    return first_word


def simple_normalize(text, reference_options=None, mcq_mapping=None):
    """
    Normalize text for comparison, with special handling for MCQ answers.
    
    Args:
        text: The text to normalize (could be a letter or full answer text)
        reference_options: List of MCQ options (for old template compatibility)
        mcq_mapping: Dict mapping letters to full option text (e.g., {"A": "Paris", "B": "Lyon"})
    
    Returns:
        Normalized text or empty string if text is empty/None
    """
    logger.debug(f"simple_normalize called with text='{text}', mcq_mapping={mcq_mapping}")
    
    # Handle None or empty cases first
    if text is None or text == "":
        logger.debug("simple_normalize: Received None or empty string")
        return ""
    
    if isinstance(text, bool):
        result = "نعم" if text else "لا"
        logger.debug(f"simple_normalize: Boolean converted to '{result}'")
        return result
    
    if not isinstance(text, str):
        text = str(text)
        logger.debug(f"simple_normalize: Converted to string: '{text}'")
    
    # Check again after conversion
    if not text.strip():
        logger.debug("simple_normalize: Empty string after strip")
        return ""
    
    extracted = extract_first_word_or_line(text)
    clean_extracted = re.sub(r"[()[\]{}]", "", extracted).strip()
    logger.debug(f"simple_normalize: extracted='{extracted}', clean_extracted='{clean_extracted}'")

    # If we have an MCQ mapping (new template), handle letter-to-text conversion
    if mcq_mapping:
        logger.debug("simple_normalize: Using MCQ mapping")
        
        # Check if the extracted text is a single letter (A, B, C, D, etc.)
        if len(extracted.strip()) == 1 and extracted.strip().upper().isalpha():
            letter = extracted.strip().upper()
            logger.debug(f"simple_normalize: Detected single letter: '{letter}'")
            # Convert letter to full text using mapping
            if letter in mcq_mapping:
                result = mcq_mapping[letter]
                logger.info(f"simple_normalize: Mapped letter '{letter}' to option '{result}'")
                return result
        
        # Check for "A)" format
        mc_match = re.match(r"^([A-Za-z])\)", extracted.strip())
        if mc_match:
            letter = mc_match.group(1).upper()
            logger.debug(f"simple_normalize: Detected letter with parenthesis: '{letter})'")
            if letter in mcq_mapping:
                result = mcq_mapping[letter]
                logger.info(f"simple_normalize: Mapped '{letter})' to option '{result}'")
                return result
        
        # If it's already full text, return it normalized
        text_lower = extracted.lower().strip()
        for letter, option_text in mcq_mapping.items():
            if text_lower == option_text.lower().strip():
                logger.info(f"simple_normalize: Matched full text to option '{option_text}'")
                return option_text
        
        logger.debug(f"simple_normalize: No MCQ match found, returning '{clean_extracted or extracted}'")
        return clean_extracted or extracted

    # Old template compatibility (reference_options provided)
    logger.debug("simple_normalize: Using old template (reference_options)")
    
    mc_match = re.match(r"^([A-Za-z])\)", extracted.strip())
    if mc_match:
        letter = mc_match.group(1)
        logger.debug(f"simple_normalize: Found letter pattern '{letter})'")
        if reference_options:
            for option in reference_options:
                if option.strip().upper() == letter.upper():
                    logger.info(f"simple_normalize: Matched letter '{letter}' to option '{option}'")
                    return option
        return letter
    
    if len(extracted.strip()) == 1 and extracted.strip().isalpha():
        logger.debug(f"simple_normalize: Single letter answer: '{extracted.strip()}'")
        return extracted.strip()
    
    if reference_options:
        extracted_lower = extracted.lower().strip()
        clean_lower = clean_extracted.lower().strip()
        for option in reference_options:
            opt_low = option.lower().strip()
            if extracted_lower == opt_low or clean_lower == opt_low:
                logger.info(f"simple_normalize: Exact match to option '{option}'")
                return option
            if (
                len(option.strip()) == 1
                and extracted.strip().upper().startswith(option.strip().upper())
            ):
                logger.info(f"simple_normalize: Prefix match to option '{option}'")
                return option
            clean_opt = re.sub(r"[()[\]{}]", "", opt_low)
            if clean_opt in clean_lower or clean_lower in clean_opt:
                logger.info(f"simple_normalize: Fuzzy match to option '{option}'")
                return option
    
    result = clean_extracted or extracted
    logger.debug(f"simple_normalize: Returning default '{result}'")
    return result


# ---------- aggregation (register if missing) ----------
if "accuracy" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("accuracy")
    def accuracy_aggregation(items):
        logger.info(f"accuracy_aggregation: Processing {len(items)} items")
        correct = 0
        total = len(items)
        
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                ref = item.get("ref", "")
                pred = item.get("pred", "")
                mcq_mapping = item.get("mcq_mapping")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                ref = item[0]
                pred = item[1]
                mcq_mapping = item[2] if len(item) >= 3 else None
            else:
                logger.warning(f"accuracy_aggregation: Skipping invalid item at index {idx}: {type(item)}")
                continue
            
            # Handle empty predictions - count as incorrect
            if pred is None or (isinstance(pred, str) and not pred.strip()):
                logger.warning(f"accuracy_aggregation [{idx}]: Empty prediction, counting as incorrect")
                continue
            
            logger.debug(f"accuracy_aggregation [{idx}]: ref='{ref}', pred='{pred}'")
            
            # Normalize both reference and prediction
            norm_ref = simple_normalize(ref, mcq_mapping=mcq_mapping)
            norm_pred = simple_normalize(pred, mcq_mapping=mcq_mapping)
            
            # If normalization resulted in empty strings, skip
            if not norm_ref or not norm_pred:
                logger.warning(f"accuracy_aggregation [{idx}]: Normalization resulted in empty string(s), counting as incorrect")
                continue
            
            logger.debug(f"accuracy_aggregation [{idx}]: norm_ref='{norm_ref}', norm_pred='{norm_pred}'")
            
            # Clean for comparison
            clean_ref = re.sub(r"[()[\]{}]", "", norm_ref).lower().strip()
            clean_pred = re.sub(r"[()[\]{}]", "", norm_pred).lower().strip()
            
            logger.debug(f"accuracy_aggregation [{idx}]: clean_ref='{clean_ref}', clean_pred='{clean_pred}'")
            
            if clean_ref == clean_pred:
                correct += 1
                logger.info(f"accuracy_aggregation [{idx}]: CORRECT ✓")
            else:
                logger.info(f"accuracy_aggregation [{idx}]: INCORRECT ✗ (expected '{clean_ref}', got '{clean_pred}')")
        
        accuracy = correct / total if total else 0.0
        logger.info(f"accuracy_aggregation: Final accuracy = {correct}/{total} = {accuracy:.4f}")
        return accuracy


# ---------- metric (register if missing) ----------
if "accuracy" not in le_registry.METRIC_REGISTRY:
    @register_metric(
        metric="accuracy",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="accuracy",
    )
    def accuracy_fn(items):
        logger.debug(f"accuracy_fn: Received {len(items) if isinstance(items, list) else 1} items")
        return items


def process_results(doc, results):
    """
    Process results and create MCQ mapping if available.
    """
    logger.debug(f"process_results: doc keys={list(doc.keys())}, results type={type(results)}")
    
    # Extract prediction with better empty handling
    if isinstance(results, str):
        pred = results.strip()
    elif isinstance(results, list) and results:
        pred = str(results[0]).strip() if results[0] is not None else ""
    elif isinstance(results, dict):
        raw_pred = results.get("text", results.get("generated_text", results.get("prediction", "")))
        pred = str(raw_pred).strip() if raw_pred is not None else ""
    else:
        pred = str(results).strip() if results is not None else ""
    
    # Ensure pred is not None
    if pred is None:
        pred = ""
    
    logger.debug(f"process_results: Extracted prediction='{pred}'")
    
    ref = doc.get("output", "")
    if isinstance(ref, list) and ref:
        ref = str(ref[0]) if ref[0] is not None else ""
    elif ref is None:
        ref = ""
    
    logger.debug(f"process_results: Reference='{ref}'")
    
    # Create MCQ mapping if MCQ options exist (new template)
    mcq_mapping = None
    if "mcq" in doc and isinstance(doc.get("mcq"), list):
        mcq_options = doc["mcq"]
        # Create letter-to-text mapping (A, B, C, D, etc.)
        letters = [chr(65 + i) for i in range(len(mcq_options))]
        mcq_mapping = {letter: option for letter, option in zip(letters, mcq_options)}
        logger.info(f"process_results: Created MCQ mapping with {len(mcq_mapping)} options: {mcq_mapping}")
    else:
        logger.debug("process_results: No MCQ options found (old template or non-MCQ question)")
    
    result = {"accuracy": [ref, pred, mcq_mapping]}
    logger.debug(f"process_results: Returning {result}")
    return result


# ---------- Helper function to detect template version ----------
def is_new_template(doc: dict) -> bool:
    """
    Detect if document uses new template format (mcq key exists).
    Returns True for new template, False for old template.
    """
    is_new = "mcq" in doc and isinstance(doc.get("mcq"), list)
    logger.debug(f"is_new_template: {is_new} (mcq key exists: {'mcq' in doc})")
    return is_new


# ---------- Helper function to format MCQ options ----------
def format_mcq_options(mcq_options: list) -> str:
    """
    Format MCQ options with letter prefixes (A, B, C, D, etc.)
    
    Args:
        mcq_options: List of option texts
        
    Returns:
        Formatted string with options labeled A, B, C, etc.
    """
    if not mcq_options:
        logger.debug("format_mcq_options: No options provided")
        return ""
    
    # Use uppercase letters A, B, C, D, etc.
    letters = [chr(65 + i) for i in range(len(mcq_options))]
    formatted_options = []
    
    for letter, option in zip(letters, mcq_options):
        formatted_options.append(f"{letter} - {option}")
    
    result = "\n".join(formatted_options)
    logger.debug(f"format_mcq_options: Formatted {len(mcq_options)} options")
    return result


# ---------- BaseMetric for YAML ----------
class AccuracyMetric(BaseMetric):
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """
        Returns a function that dynamically formats doc_to_text based on template version.
        This allows the metric to adapt to both old and new templates.
        """
        logger.debug("AccuracyMetric.get_doc_to_text: Returning Jinja2 template")
        # Return a template that will be processed by Jinja2
        # Use double quotes to avoid YAML escaping issues with single quotes
        return '''{{ instruction }}
{% if input is string %}
{{ input }}
{% else %}
{{ input|join("\\n") }}
{% endif %}

{% if mcq is defined and mcq|length > 0 %}
{% set letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] %}
{% for option in mcq %}{{ letters[loop.index0] }} - {{ option }}
{% endfor %}

تعليمات الإخراج:
أعد رمز الخيار المختار فقط كما هو مذكور في قائمة الخيارات، دون كتابة نص الخيار، ودون أي شرح أو تعليق إضافي.
{% else %}
تعليمات:
- أجب بحرف الخيار فقط (A,B,C,D) إن وُجد.
- أو بكلمة واحدة من الخيارات.
{% endif %}
الإجابة:'''

    def get_generation_kwargs(self):
        kwargs = {
            "do_sample": False,
            "until": [".", "،", "؟", "!"],
            "max_gen_toks": 4096,
        }
        logger.debug(f"AccuracyMetric.get_generation_kwargs: {kwargs}")
        return kwargs


# ---------- self-registration for YAML export ----------
logger.info("Initializing accuracy metric configuration")
config = MetricConfig(
    name="accuracy",
    higher_is_better=True,
    aggregation_name="accuracy",
    process_results=process_results,
)
get_metrics_registry().register("accuracy", AccuracyMetric(config))
logger.info("Accuracy metric registered successfully")