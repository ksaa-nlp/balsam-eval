"""Registry system for metrics (auto-register enabled)."""

from typing import Any, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

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
        return list(self._metrics.keys())

    def detect_metric_type(self, metric_name: str) -> Optional[str]:
        metric_name_lower = metric_name.lower()
        for registered in self._metrics.keys():
            if registered in metric_name_lower:
                return registered
        return None


# ---------------- Global singleton ----------------
_registry: Optional[MetricsRegistry] = None

def get_metrics_registry() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry
