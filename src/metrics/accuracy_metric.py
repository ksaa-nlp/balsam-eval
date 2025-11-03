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
others = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
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


def simple_normalize(text, reference_options=None):
    if isinstance(text, bool):
        return "نعم" if text else "لا"
    if not isinstance(text, str):
        text = str(text)
    extracted = extract_first_word_or_line(text)
    clean_extracted = re.sub(r"[()[\]{}]", "", extracted).strip()

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
                ref, pred = item.get("ref", ""), item.get("pred", "")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                ref, pred = item[0], item[1]
            else:
                continue
            norm_ref = simple_normalize(ref)
            norm_pred = simple_normalize(pred)
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
    return {"accuracy": [ref, pred]}


# ---------- BaseMetric for YAML ----------
class AccuracyMetric(BaseMetric):
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return (
            f"{original_doc_to_text}\n\n"
            "تعليمات:\n- أجب بحرف الخيار فقط (A,B,C,D) إن وُجد.\n"
            "- أو بكلمة واحدة من الخيارات.\n"
            "الإجابة:"
        )

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
