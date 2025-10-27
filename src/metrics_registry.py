"""Registry system for metrics."""

from typing import Any, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .metrics import bleu_metric, rouge_metric, accuracy_metric

@dataclass
class MetricConfig:
    """Configuration for a metric."""
    name: str
    aggregation_name: Optional[str] = None
    higher_is_better: bool = True
    output_type: str = "generate_until"
    generation_kwargs: Optional[Dict[str, Any]] = None
    process_results: Optional[Callable[[Any], Any]] = None


class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, config: MetricConfig):
        self.config = config
    
    @abstractmethod
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        """Modify doc_to_text if needed."""
        pass
    
    @abstractmethod
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs."""
        pass
    
    def get_yaml_config(self, base_yaml: Dict[str, Any]) -> Dict[str, Any]:
        """Get complete YAML configuration."""
        yaml_config = base_yaml.copy()
        
        # Update doc_to_text
        original_doc_to_text = yaml_config.get("doc_to_text", "{{instruction}}\n{{input}}")
        yaml_config["doc_to_text"] = self.get_doc_to_text(original_doc_to_text)
        
        # Update generation kwargs
        yaml_config["generation_kwargs"] = self.get_generation_kwargs()
        if self.config.process_results is not None:
            yaml_config["process_results"] = self.config.process_results
        
        # Add metric config
        yaml_config.update({
            "output_type": self.config.output_type,
            "metric_list": [{
                "metric": self.config.name,
                "aggregation": self.config.aggregation_name if self.config.aggregation_name is not None else self.config.name,
                "higher_is_better": self.config.higher_is_better,
            }],
        })
        
        return yaml_config


class RougeMetric(BaseMetric):
    """ROUGE metric."""
    
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return original_doc_to_text
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        return {
            "do_sample": False,
            "until": ["<|endoftext|>"],
        }


class BleuMetric(BaseMetric):
    """BLEU metric."""
    
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return original_doc_to_text
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        return {
            "do_sample": False,
            "until": ["<|endoftext|>"],
        }


class AccuracyMetric(BaseMetric):
    """Accuracy metric with enhanced instructions."""
    
    def get_doc_to_text(self, original_doc_to_text: str) -> str:
        return (
            f"{original_doc_to_text}\n\n"
            "تعليمات مهمة:\n"
            "- اقرأ السؤال بعناية وحدد الخيارات المتاحة.\n"
            "- إذا كانت الخيارات تحتوي على حروف (أ، ب، ج، د أو A، B، C، D)، أجب بالحرف فقط.\n"
            "- إذا لم تكن هناك حروف، أجب بكلمة واحدة فقط من الخيارات المذكورة.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Read the question carefully and identify the available options.\n"
            "- If options contain letters (A, B, C, D), answer with ONLY the letter.\n"
            "- If there are no letter options, answer with ONLY ONE WORD from the options.\n\n"
            "الإجابة:"
        )
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        return {
            "do_sample": False,
            "until": ["<|endoftext|>", "\n", ".", "،", "؟", "!", " "],
            "max_gen_toks": 5,
        }


class MetricsRegistry:
    """Registry for all metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, BaseMetric] = {}
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register default metrics."""
        
        # BLEU
        bleu_config = MetricConfig(name="bleu", higher_is_better=True, aggregation_name="custom_bleu", process_results=bleu_metric.process_results)
        self._metrics["bleu"] = BleuMetric(bleu_config)
        
        # ROUGE
        rouge_config = MetricConfig(name="rouge", higher_is_better=True,process_results=rouge_metric.process_results)
        self._metrics["rouge"] = RougeMetric(rouge_config)
        
        # Accuracy
        accuracy_config = MetricConfig(name="accuracy", higher_is_better=True, process_results=accuracy_metric.process_results)
        self._metrics["accuracy"] = AccuracyMetric(accuracy_config)
    
    def register(self, name: str, metric: BaseMetric):
        """Register a new metric."""
        self._metrics[name.lower()] = metric
    
    def get(self, name: str) -> Optional[BaseMetric]:
        """Get a metric by name."""
        return self._metrics.get(name.lower())
    
    def list_metrics(self) -> list[str]:
        """List all registered metrics."""
        return list(self._metrics.keys())
    
    def detect_metric_type(self, metric_name: str) -> Optional[str]:
        """Detect metric type from name."""
        metric_name_lower = metric_name.lower()
        
        for registered_name in self._metrics.keys():
            if registered_name in metric_name_lower:
                return registered_name
        
        return None


# Global registry
_registry = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry