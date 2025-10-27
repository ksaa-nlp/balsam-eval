# Adding a New Metric

This guide explains how to add a new evaluation metric to the Balsam Evaluation framework.

## Overview

The metrics system in Balsam-eval is designed to be extensible. Each metric is implemented as a separate module in the `src/metrics` directory and follows a specific pattern for registration and implementation.

## Step-by-Step Guide

### 1. Create a New Metric File

Create a new Python file in the `src/metrics` directory. Name it according to your metric (e.g., `my_metric.py`).

```python
"""[YOUR_METRIC_NAME] metric implementation."""

from lm_eval.api.registry import register_aggregation, register_metric

# Import any required dependencies
```

### 2. Implement the Metric

Your metric implementation should include:

1. Any necessary preprocessing functions
2. An aggregation function decorated with `@register_aggregation`

Example structure:

```python
def preprocess_text(text):
    """Preprocess text for evaluation."""
    # Your preprocessing logic here
    return processed_text

@register_aggregation("your_metric_name")
def your_metric_aggregation(items):
    """Aggregate your metric scores across the dataset."""
    refs = list(zip(*items))[0]  # Reference texts
    preds = list(zip(*items))[1]  # Model predictions
    
    # Your metric computation logic here
    
    return {
        "score": final_score,
        # Add any additional metrics you want to track
    }
```

### 3. Register the Metric

1. Add your metric module to `src/metrics/__init__.py`:

```python
from . import your_metric_module

__all__ = [..., 'your_metric_module']
```

2. Create a metric configuration in your code where you use the metric:

```python
from src.metrics_registry import MetricConfig

metric_config = MetricConfig(
    name="your_metric_name",
    aggregation_name="your_metric_name",  # Same as used in @register_aggregation
    higher_is_better=True,  # Set to True if higher scores are better
    output_type="generate_until",  # Type of generation required
    generation_kwargs={},  # Any specific generation parameters
    process_results=None  # Optional function to process results
)
```

### 4. Implement Required Base Class Methods

If your metric requires special handling of document text or generation parameters, implement the necessary methods from `BaseMetric`:

```python
from src.metrics_registry import BaseMetric

class YourMetric(BaseMetric):
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Modify doc_to_text if needed."""
        return original_doc_to_text
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs."""
        return {
            # Your generation parameters
        }
```

## Best Practices

1. **Documentation**: Include detailed docstrings explaining your metric's purpose and implementation.
2. **Preprocessing**: Include any necessary text preprocessing steps in your metric module.
3. **Error Handling**: Add appropriate error handling for edge cases.
4. **Type Hints**: Use Python type hints for better code maintainability.
5. **Testing**: Add unit tests for your metric in the tests directory.

## Example

Here's a minimal example of a complete metric implementation:

```python
"""Example metric implementation."""

from lm_eval.api.registry import register_aggregation

def normalize_text(text):
    """Normalize text for comparison."""
    return text.strip().lower()

@register_aggregation("example_metric")
def example_aggregation(items):
    """Calculate example metric score."""
    refs = [normalize_text(ref) for ref, _ in items]
    preds = [normalize_text(pred) for _, pred in items]
    
    # Calculate your metric
    scores = [
        1 if ref == pred else 0
        for ref, pred in zip(refs, preds)
    ]
    
    return {
        "score": sum(scores) / len(scores),
        "num_samples": len(scores)
    }
```

## Adding to the Registry

After implementing your metric, make sure it's properly registered by updating the imports in `src/metrics_registry.py`:

```python
from .metrics import your_metric_module
```

## Testing Your Metric

1. Create test cases in the tests directory
2. Run the tests to ensure your metric works as expected
3. Test with a small dataset before using in production

For any additional help or specific use cases, please refer to the existing metric implementations in the `src/metrics` directory as examples.
