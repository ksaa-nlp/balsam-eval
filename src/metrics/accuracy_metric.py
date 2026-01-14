import re
import unicodedata
import sys
from lm_eval.api.registry import register_aggregation, register_metric
from lm_eval.api import registry as le_registry
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

# ---------- punctuation setup ----------
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"—ـ"""
ALL_PUNCTUATIONS += "".join([o for o in others if o not in ALL_PUNCTUATIONS])


# ---------- helpers ----------
def extract_first_word_or_line(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""
    first_line = text.split("\n")[0].strip()
    first_line = re.sub(r"[.،؟!]+$", "", first_line)
    if len(first_line.split()) <= 3:
        return first_line
    first_word = first_line.split()[0] if first_line.split() else first_line
    first_word = re.sub(r'^["\'""]|["\'""]$', "", first_word)
    first_word = re.sub(r"[.،؟!]+$", "", first_word)
    return first_word


def simple_normalize(text, reference_options=None, mcq_mapping=None):
    """
    Normalize text for comparison, with special handling for MCQ answers.
    
    Args:
        text: The text to normalize (could be a letter or full answer text)
        reference_options: List of MCQ options (for old template compatibility)
        mcq_mapping: Dict mapping letters to full option text (e.g., {"A": "Paris", "B": "Lyon"})
    
    Returns:
        Normalized text
    """
    if isinstance(text, bool):
        return "نعم" if text else "لا"
    if not isinstance(text, str):
        text = str(text)
    
    extracted = extract_first_word_or_line(text)
    clean_extracted = re.sub(r"[()[\]{}]", "", extracted).strip()

    # If we have an MCQ mapping (new template), handle letter-to-text conversion
    if mcq_mapping:
        # Check if the extracted text is a single letter (A, B, C, D, etc.)
        if len(extracted.strip()) == 1 and extracted.strip().upper().isalpha():
            letter = extracted.strip().upper()
            # Convert letter to full text using mapping
            if letter in mcq_mapping:
                return mcq_mapping[letter]
        
        # Check for "A)" format
        mc_match = re.match(r"^([A-Za-z])\)", extracted.strip())
        if mc_match:
            letter = mc_match.group(1).upper()
            if letter in mcq_mapping:
                return mcq_mapping[letter]
        
        # If it's already full text, return it normalized
        text_lower = extracted.lower().strip()
        for letter, option_text in mcq_mapping.items():
            if text_lower == option_text.lower().strip():
                return option_text
        
        return clean_extracted or extracted

    # Old template compatibility (reference_options provided)
    mc_match = re.match(r"^([A-Za-z])\)", extracted.strip())
    if mc_match:
        letter = mc_match.group(1)
        if reference_options:
            for option in reference_options:
                if option.strip().upper() == letter.upper():
                    return option
        return letter
    
    if len(extracted.strip()) == 1 and extracted.strip().isalpha():
        return extracted.strip()
    
    if reference_options:
        extracted_lower = extracted.lower().strip()
        clean_lower = clean_extracted.lower().strip()
        for option in reference_options:
            opt_low = option.lower().strip()
            if extracted_lower == opt_low or clean_lower == opt_low:
                return option
            if (
                len(option.strip()) == 1
                and extracted.strip().upper().startswith(option.strip().upper())
            ):
                return option
            clean_opt = re.sub(r"[()[\]{}]", "", opt_low)
            if clean_opt in clean_lower or clean_lower in clean_opt:
                return option
    
    return clean_extracted or extracted


# ---------- aggregation (register if missing) ----------
if "accuracy" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("accuracy")
    def accuracy_aggregation(items):
        correct = 0
        total = len(items)
        for item in items:
            if isinstance(item, dict):
                ref = item.get("ref", "")
                pred = item.get("pred", "")
                mcq_mapping = item.get("mcq_mapping")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                ref = item[0]
                pred = item[1]
                mcq_mapping = item[2] if len(item) >= 3 else None
            else:
                continue
            
            # Normalize both reference and prediction
            norm_ref = simple_normalize(ref, mcq_mapping=mcq_mapping)
            norm_pred = simple_normalize(pred, mcq_mapping=mcq_mapping)
            
            # Clean for comparison
            clean_ref = re.sub(r"[()[\]{}]", "", norm_ref).lower().strip()
            clean_pred = re.sub(r"[()[\]{}]", "", norm_pred).lower().strip()
            
            if clean_ref == clean_pred:
                correct += 1
        
        return correct / total if total else 0.0


# ---------- metric (register if missing) ----------
if "accuracy" not in le_registry.METRIC_REGISTRY:
    @register_metric(
        metric="accuracy",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="accuracy",
    )
    def accuracy_fn(items):
        return items


def process_results(doc, results):
    """
    Process results and create MCQ mapping if available.
    """
    if isinstance(results, str):
        pred = results.strip()
    elif isinstance(results, list) and results:
        pred = str(results[0]).strip()
    elif isinstance(results, dict):
        pred = str(results.get("text", results.get("generated_text", results.get("prediction", "")))).strip()
    else:
        pred = str(results).strip()
    
    ref = doc.get("output", "")
    if isinstance(ref, list) and ref:
        ref = str(ref[0])
    
    # Create MCQ mapping if MCQ options exist (new template)
    mcq_mapping = None
    if "mcq" in doc and isinstance(doc.get("mcq"), list):
        mcq_options = doc["mcq"]
        # Create letter-to-text mapping (A, B, C, D, etc.)
        letters = [chr(65 + i) for i in range(len(mcq_options))]
        mcq_mapping = {letter: option for letter, option in zip(letters, mcq_options)}
    
    return {"accuracy": [ref, pred, mcq_mapping]}


# ---------- Helper function to detect template version ----------
def is_new_template(doc: dict) -> bool:
    """
    Detect if document uses new template format (mcq key exists).
    Returns True for new template, False for old template.
    """
    return "mcq" in doc and isinstance(doc.get("mcq"), list)


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
        return ""
    
    # Use uppercase letters A, B, C, D, etc.
    letters = [chr(65 + i) for i in range(len(mcq_options))]
    formatted_options = []
    
    for letter, option in zip(letters, mcq_options):
        formatted_options.append(f"{letter} - {option}")
    
    return "\n".join(formatted_options)


# ---------- BaseMetric for YAML ----------
class AccuracyMetric(BaseMetric):
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """
        Returns a function that dynamically formats doc_to_text based on template version.
        This allows the metric to adapt to both old and new templates.
        """
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
        return {
            "do_sample": False,
            "until": ["<|endoftext|>", "\n", ".", "،", "؟", "!", " "],
            "max_gen_toks": 5,
        }


# ---------- self-registration for YAML export ----------
config = MetricConfig(
    name="accuracy",
    higher_is_better=True,
    aggregation_name="accuracy",
    process_results=process_results,
)
get_metrics_registry().register("accuracy", AccuracyMetric(config))