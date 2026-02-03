
````markdown
# 🧮 Custom Metrics Guide

This folder contains all evaluation metrics used by **balsam-eval**.

Each metric file:
- Lives in this folder (`src/metrics/`)
- Ends with `_metric.py` (e.g. `rouge_metric.py`, `bleu_metric.py`, `new_metric.py`)
- Automatically registers itself in:
  - **lm-eval** registry (`@register_metric`, `@register_aggregation`)
  - **our internal registry** (`src/metrics_registry.py`)

---

## 📦 1. Metric file structure

Use this base layout when adding a new metric.

```python
from lm_eval.api.registry import register_metric, register_aggregation
from lm_eval.api import registry as le_registry
from src.metrics_registry import BaseMetric, MetricConfig, get_metrics_registry

# ---- aggregation function ----
if "my_metric_agg" not in le_registry.AGGREGATION_REGISTRY:
    @register_aggregation("my_metric_agg")
    def my_metric_aggregation(items):
        """Combine multiple results into one score."""
        # TODO: compute aggregated value
        return 0.0

# ---- metric registration ----
if "my_metric" not in le_registry.METRIC_REGISTRY:
    @register_metric(
        metric="my_metric",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="my_metric_agg",
    )
    def my_metric(items):
        """Return raw items for aggregation."""
        return items

# ---- process_results ----
def process_results(doc, results):
    """Extract reference & prediction from dataset sample."""
    # TODO: fill extraction logic
    return {"my_metric": ["", ""]}

# ---- YAML export integration ----
class MyMetric(BaseMetric):
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return original_doc_to_text

    def get_generation_kwargs(self):
        return {}

# ---- custom registry registration ----
config = MetricConfig(
    name="my_metric",
    aggregation_name="my_metric_agg",
    higher_is_better=True,
    process_results=process_results,
)
get_metrics_registry().register("my_metric", MyMetric(config))
````

---

## 🧩 2. Naming rules

| Part                 | Description                                   | Example                                           |
| -------------------- | --------------------------------------------- | ------------------------------------------------- |
| **File name**        | Must end with `_metric.py`                    | `rouge_metric.py`                                 |
| **Metric name**      | Unique key in both registries                 | `my_metric`                                       |
| **Aggregation name** | Distinct from built-in ones to avoid conflict | `my_metric_agg`                                   |
| **YAML task config** | Must reference both                           | `metric: my_metric`, `aggregation: my_metric_agg` |

---

## 🧠 3. How it connects

| Layer                                        | Purpose                                           |
| -------------------------------------------- | ------------------------------------------------- |
| `@register_metric` / `@register_aggregation` | Exposes metric to **lm-eval** runtime             |
| `process_results()`                          | Defines how predictions and references are paired |
| `BaseMetric` subclass                        | Adds YAML generation support                      |
| `metrics_registry.register()`                | Adds it to our internal registry for discovery    |

---

## ⚙️ 4. Testing locally

1. Place your new file in `src/metrics/`.

2. Run:

   ```bash
   python -m src.metrics.new_metric
   ```

   or run the full evaluation:

   ```bash
   python run_local.py
   ```

3. Ensure it appears in the registry:

   ```python
   from src.metrics_registry import get_metrics_registry
   print(get_metrics_registry().list_metrics())
   ```

---

## ✅ 5. Common mistakes to avoid

* ❌ Forgetting `_metric.py` suffix → won’t be auto-imported
* ❌ Reusing a built-in metric name (e.g. `"bleu"`) → causes duplicate registry error
* ❌ Omitting `aggregation:` in YAML → causes `KeyError: None` in `lm_eval`
* ✅ Always use unique names like `my_metric`, `my_metric_agg`

---

### Example YAML snippet

```yaml
metric_list:
  - metric: my_metric
    aggregation: my_metric_agg
    higher_is_better: true
```

---

> 🧩 Tip: To quickly start, copy `src/metrics/new_metric.py` (the empty template file) and rename all `"new_metric"` occurrences to your metric’s name.
