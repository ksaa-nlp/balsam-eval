import re
import unicodedata
import sys
import evaluate
from lm_eval.api.registry import register_aggregation, register_metric
from lm_eval.api import registry as le_registry
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

rouge = evaluate.load("rouge")
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in others if o not in ALL_PUNCTUATIONS])

def prepare_texts(text, change_curly_braces=False):
    text = re.sub("([" + ALL_PUNCTUATIONS + "])", " \\1 ", text)
    if change_curly_braces:
        text = text.replace("{", "[").replace("}", "]")
    return text

# ---- make sure an aggregation named "rouge" exists (lm-eval sometimes lacks default)
if "rouge" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("rouge")
    def rouge_aggregation(items):
        def tokenizer(x): return x.split()
        refs = [prepare_texts(r, False).strip() for r, _ in items]
        preds = [prepare_texts(p, True).strip() for _, p in items]

        total = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
        for ref, pred in zip(refs, preds):
            score = rouge.compute(references=[ref], predictions=[pred], tokenizer=tokenizer)
            for k in total.keys():
                total[k] += score[k]
        count = len(refs) if refs else 1
        return {k: v / count for k, v in total.items()}

# ---- metric "rouge" (register only if missing)
if "rouge" not in le_registry.METRIC_REGISTRY:
    @register_metric(metric="rouge", higher_is_better=True,
                     output_type="generate_until", aggregation="rouge")
    def rouge_fn(items): 
        return items

def process_results(doc, results):
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]
    return {"rouge": [golds, preds]}

class RougeMetric(BaseMetric):
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return original_doc_to_text
    def get_generation_kwargs(self):
        return {"do_sample": False, "until": ["<|endoftext|>"]}

config = MetricConfig(
    name="rouge", higher_is_better=True,
    aggregation_name="rouge", process_results=process_results)
get_metrics_registry().register("rouge", RougeMetric(config))
