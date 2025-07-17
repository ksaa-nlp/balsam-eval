import re
import evaluate
import unicodedata
import sys

# Load Rouge evaluator from `evaluate` library
rouge = evaluate.load("rouge")

# Punctuation table
PUNCT_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)
ALL_PUNCTUATIONS = "".join(chr(p) for p in PUNCT_TABLE)
others = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
ALL_PUNCTUATIONS += "".join([o for o in others if o not in ALL_PUNCTUATIONS])


def prepare_texts(text, change_curly_braces):

    # 1. put spaces before and after each punctuation
    text = re.sub("([" + ALL_PUNCTUATIONS + "])", " \\1 ", text)

    # 2. change all {} to []
    if change_curly_braces:
        text = text.replace("{", "[").replace("}", "]")

    return text


def rouge1_scores(predictions, references):  # This is a passthrough function

    return (predictions[0], references[0])


def custom_rouge_agg(items):
    def tokenizer(x): return x.split()
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    # Preprocess the texts
    refs = [prepare_texts(ref, change_curly_braces=False).strip()
            for ref in refs]
    preds = [prepare_texts(pred, change_curly_braces=True).strip()
             for pred in preds]
    # Initialize sums for each ROUGE score
    rouge1 = 0.0
    rouge2 = 0.0
    rougeL = 0.0
    rougeLsum = 0.0

    for i in range(len(refs)):
        # Compute ROUGE scores
        score = rouge.compute(
            references=[refs[i]], predictions=[preds[i]], tokenizer=tokenizer
        )

        # Aggregate each type of ROUGE score
        rouge1 += score["rouge1"]
        rouge2 += score["rouge2"]
        rougeL += score["rougeL"]
        rougeLsum += score["rougeLsum"]

    count = len(refs)
    avg_rouge1 = rouge1 / count
    avg_rouge2 = rouge2 / count
    avg_rougeL = rougeL / count
    avg_rougeLsum = rougeLsum / count

    return {
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "rougeLsum": avg_rougeLsum,
    }


def process_results(doc, results):
    preds, golds = results[0], doc["output"]
    # rouge_scores = evaluate_rouge_metric(preds, golds)
    # print(f"ROUGE Scores: {rouge_scores}")
    return {"rouge": [golds, preds]}


# A function to normalize strings to id like format (normalized and sanitized to be used as a file name too)
def normalize_string(text: str) -> str:
    return (
        unicodedata.normalize("NFKC", text)
        .lower()
        .replace("\x00", "")
        .strip()
        [:255]
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
