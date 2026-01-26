"""Registry system for metrics (auto-register enabled)."""

from typing import Any, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from lm_eval.api.registry import METRIC_REGISTRY, AGGREGATION_REGISTRY


# ---------------- Metric Config ----------------
@dataclass
class MetricConfig:
    """Configuration describing one metric."""
    name: str
    aggregation_name: Optional[str] = None
    higher_is_better: bool = True
    output_type: str = "generate_until"
    generation_kwargs: Optional[Dict[str, Any]] = None
    process_results: Optional[Callable[[Any], Any]] = None


# ---------------- Base Metric ----------------
class BaseMetric(ABC):
    """Base class that each metric can extend for YAML/task export."""

    def __init__(self, config: MetricConfig):
        self.config = config

    @abstractmethod
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        pass

    @abstractmethod
    def get_generation_kwargs(self) -> Dict[str, Any]:
        pass

    def get_yaml_config(self, base_yaml: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base YAML with metric-specific configuration."""
        yaml_config = base_yaml.copy()

        # update doc_to_text
        original_doc_to_text = yaml_config.get("doc_to_text", "{{instruction}}\n{{input}}")
        yaml_config["doc_to_text"] = self.get_doc_to_text(original_doc_to_text)

        # generation args
        yaml_config["generation_kwargs"] = self.get_generation_kwargs()

        if self.config.process_results is not None:
            yaml_config["process_results"] = self.config.process_results

        yaml_config.update({
            "output_type": self.config.output_type,
            "metric_list": [{
                "metric": self.config.name,
                "aggregation": self.config.aggregation_name or self.config.name,
                "higher_is_better": self.config.higher_is_better,
            }],
        })
        return yaml_config


# ---------------- Registry ----------------
class MetricsRegistry:
    """Global registry holding all metric classes."""
    def __init__(self):
        self._metrics: Dict[str, BaseMetric] = {}

    def register(self, name: str, metric: BaseMetric):
        self._metrics[name.lower()] = metric

    def get(self, name: str) -> Optional[BaseMetric]:
        return self._metrics.get(name.lower())

    def list_metrics(self) -> list[str]:
        """List all available metrics (custom + LM Harness)."""
        custom = list(self._metrics.keys())
        lm_harness = list(METRIC_REGISTRY.keys())
        return custom + lm_harness

    def is_lm_harness_metric(self, metric_name: str) -> bool:
        """Check if metric is a built-in LM Harness metric."""
        return metric_name.lower() in [k.lower() for k in METRIC_REGISTRY.keys()]

    def detect_metric_type(self, metric_name: str) -> Optional[str]:
        """Detect metric type with priority: custom metrics > LM Harness metrics."""
        if not metric_name:
            return None
            
        metric_name_lower = metric_name.lower()
        
        # First check custom metrics (exact match) - CASE INSENSITIVE
        for registered in self._metrics.keys():
            if registered.lower() == metric_name_lower:
                return registered  # Return the actual registered name
        
        # Check custom metrics (partial match)
        for registered in self._metrics.keys():
            if registered in metric_name_lower or metric_name_lower in registered:
                return registered
        
        # Then check LM Harness metrics (exact match)
        for registered in METRIC_REGISTRY.keys():
            if registered.lower() == metric_name_lower:
                return registered
        
        # Check LM Harness metrics (partial match)
        for registered in METRIC_REGISTRY.keys():
            if registered.lower() in metric_name_lower or metric_name_lower in registered.lower():
                return registered
        
        return None
    
    def get_metric_info(self, metric_name: str) -> Dict[str, Any]:
        """Get information about a metric."""
        metric_type = self.detect_metric_type(metric_name)
        
        if not metric_type:
            return {
                "found": False,
                "name": metric_name,
                "source": None,
                "custom_metric": None,
                "lm_harness_metric": None
            }
        
        # Check if it's a custom metric
        custom_metric = self.get(metric_type)
        if custom_metric:
            return {
                "found": True,
                "name": metric_type,
                "source": "custom",
                "custom_metric": custom_metric,
                "lm_harness_metric": None,
                "higher_is_better": custom_metric.config.higher_is_better,
                "aggregation": custom_metric.config.aggregation_name or custom_metric.config.name,
                "output_type": custom_metric.config.output_type
            }
        
        # Check if it's an LM Harness metric
        if self.is_lm_harness_metric(metric_type):
            # Get metric configuration from LM Harness
            output_type, higher_is_better = self._get_lm_harness_metric_config(metric_type)
            
            # Try to get aggregation from registry
            agg_name = None
            for agg_key in AGGREGATION_REGISTRY.keys():
                if agg_key.lower() == metric_type.lower():
                    agg_name = agg_key
                    break
            
            return {
                "found": True,
                "name": metric_type,
                "source": "lm_harness",
                "custom_metric": None,
                "lm_harness_metric": True,
                "higher_is_better": higher_is_better,
                "aggregation": agg_name or metric_type,
                "output_type": output_type
            }
        
        return {
            "found": False,
            "name": metric_name,
            "source": None,
            "custom_metric": None,
            "lm_harness_metric": None
        }
    
    def _get_lm_harness_metric_config(self, metric_name: str) -> tuple[str, bool]:
        """Get output_type and higher_is_better for LM Harness metrics."""
        metric_lower = metric_name.lower()
        
        # Define configurations for common LM Harness metrics
        # Format: metric_name: (output_type, higher_is_better)
        metric_configs = {
            # Exact matching metrics
            "exact_match": ("generate_until", True),
            "em": ("generate_until", True),
            
            # F1 and related
            "f1": ("generate_until", True),
            "f1_score": ("generate_until", True),
            
            # Perplexity metrics
            "perplexity": ("loglikelihood", False),
            "ppl": ("loglikelihood", False),
            "word_perplexity": ("loglikelihood", False),
            "byte_perplexity": ("loglikelihood", False),
            "bits_per_byte": ("loglikelihood", False),
            
            # Accuracy-based
            "acc": ("loglikelihood", True),
            "acc_norm": ("loglikelihood", True),
            
            # Multiple choice
            "mc1": ("loglikelihood", True),
            "mc2": ("loglikelihood", True),
            
            # Other generation metrics
            "ter": ("generate_until", False),
            "chrf": ("generate_until", True),
            "bleu": ("generate_until", True),
            "rouge": ("generate_until", True),
            "meteor": ("generate_until", True),
            
            # Classification metrics
            "matthews_corrcoef": ("loglikelihood", True),
            "mcc": ("loglikelihood", True),
        }
        
        # Check if we have explicit config
        if metric_lower in metric_configs:
            return metric_configs[metric_lower]
        
        # Default based on metric name patterns
        if any(x in metric_lower for x in ["perplexity", "ppl", "loss"]):
            return ("loglikelihood", False)
        elif any(x in metric_lower for x in ["acc", "exact", "f1", "em"]):
            return ("generate_until", True)
        else:
            # Default to generate_until with higher_is_better=True
            return ("generate_until", True)


# ---------------- Global singleton ----------------
_registry: Optional[MetricsRegistry] = None

def get_metrics_registry() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry