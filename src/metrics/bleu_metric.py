import re
import evaluate
import pyarabic.araby as araby
import unicodedata
import sys
from lm_eval.api.registry import register_aggregation, register_metric
from lm_eval.api import registry as le_registry
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

bleu = evaluate.load("bleu")
PUNCT_TABLE = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
ALL_PUNCTUATIONS += ''.join([o for o in others if o not in ALL_PUNCTUATIONS])

def prepare_texts(text, change_curly_braces, remove_diactrics):
    text = re.sub('([' + ALL_PUNCTUATIONS + '])', ' \\1 ', text)
    if change_curly_braces:
        text = text.replace('{', '[').replace('}', ']')
    if remove_diactrics:
        text = araby.strip_diacritics(text)
    return text

# ---- aggregation name kept as in your original: "custom_bleu"
if "custom_bleu" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("custom_bleu")
    def bleu_aggregation(items):
        def tokenizer(x): return x.split()
        refs = [r[0] if isinstance(r, (list, tuple)) else r for r, _ in items]
        preds = [p for _, p in items]

        refs = [prepare_texts(r, False, True).strip() for r in refs]
        preds = [prepare_texts(p, True, True).strip() for p in preds]

        total, valid = 0.0, 0
        for ref, pred in zip(refs, preds):
            if not pred or not ref:
                continue
            try:
                score = bleu.compute(references=[ref], predictions=[pred], tokenizer=tokenizer)
                total += score["bleu"]
                valid += 1
            except ZeroDivisionError:
                continue
        return total / valid if valid else 0.0

# ---- metric "bleu" uses aggregation "custom_bleu" (register only if missing)
if "bleu" not in le_registry.METRIC_REGISTRY:
    @register_metric(metric="bleu", higher_is_better=True,
                     output_type="generate_until", aggregation="custom_bleu")
    def bleu_metric(items): 
        return items

def process_results(doc, results):
    preds, golds = results[0], doc["output"]
    return {'bleu': [golds, preds]}

class BleuMetric(BaseMetric):
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return original_doc_to_text
    def get_generation_kwargs(self):
        return {"do_sample": False, "until": [".", "،", "؟", "!"],"max_gen_toks": 4096,}

config = MetricConfig(
    name="bleu", higher_is_better=True,
    aggregation_name="custom_bleu", process_results=process_results)
get_metrics_registry().register("bleu", BleuMetric(config))
