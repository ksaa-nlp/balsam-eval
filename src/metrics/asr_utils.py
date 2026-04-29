"""Common utilities for ASR metrics (WER/CER)."""

import json
from typing import Any, Dict, List


def extract_text_from_prediction(pred: str) -> str:
    """Extract actual text content from various prediction formats.

    Handles:
    - JSON format: {"text": "..."}
    - Quoted strings: "..."
    - Plain text

    Args:
        pred: Raw prediction string

    Returns:
        Extracted text content
    """
    if not isinstance(pred, str):
        return str(pred) if pred else ""

    pred = pred.strip()

    # Try to parse as JSON first
    try:
        parsed = json.loads(pred)
        if isinstance(parsed, dict):
            # Handle {"text": "..."} format
            if "text" in parsed:
                text_value = parsed["text"]
                if isinstance(text_value, str):
                    return text_value
                return str(text_value) if text_value else ""
            # Handle other dict formats - return first string value
            for v in parsed.values():
                if isinstance(v, str):
                    return v
        elif isinstance(parsed, str):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Remove surrounding quotes if present
    if pred.startswith('"') and pred.endswith('"'):
        pred = pred[1:-1]
    elif pred.startswith("'") and pred.endswith("'"):
        pred = pred[1:-1]

    return pred


def process_results_asr(
    doc: Dict[str, Any],
    results: Any,
    metric_name: str,
) -> Dict[str, List[str]]:
    """Process results for ASR evaluation (WER/CER).

    Extracts reference and prediction from document and model results.

    Args:
        doc: Document containing reference output
        results: Model predictions (list or single value)
        metric_name: Name of the metric (wer or cer)

    Returns:
        Dictionary with metric data containing [reference, prediction]
    """
    preds = results[0] if isinstance(results, list) else results
    golds = doc["output"]

    # Extract actual text from prediction
    cleaned_pred = extract_text_from_prediction(preds)

    return {metric_name: [golds, cleaned_pred]}
