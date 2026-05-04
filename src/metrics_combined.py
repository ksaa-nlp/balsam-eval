"""Combined metrics support for multiple metric evaluation."""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CombinedMetricsRegistry:
    """Registry for combined process_results functions."""

    _functions: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, func: Callable) -> None:
        """Register a combined process_results function.

        Args:
            name: Unique name for the combined function
            func: The combined process_results function
        """
        cls._functions[name] = func
        logger.info("Registered combined function: %s", name)

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a registered combined process_results function.

        Args:
            name: Name of the combined function

        Returns:
            The combined function
        """
        if name not in cls._functions:
            raise KeyError(
                f"Combined function '{name}' not found. "
                f"Available: {list(cls._functions.keys())}"
            )
        return cls._functions[name]


def create_combined_process_results(
    metric_configs: List[Dict[str, Any]]
) -> Callable[[Dict[str, Any], Any], Dict[str, Any]]:
    """Create a combined process_results function for multiple metrics.

    Args:
        metric_configs: List of metric configurations, each containing:
            - name: Metric name
            - process_results: The process_results function (optional)

    Returns:
        A function that combines results from all metrics
    """
    # Create a callable class instance
    class CombinedFunction:
        """Callable class for combined metrics."""

        def __call__(self, doc: Dict[str, Any], results: Any) -> Dict[str, Any]:
            """Combine process_results from all custom metrics.

            Args:
                doc: Document containing reference output
                results: Model predictions

            Returns:
                Dictionary with combined metric data from all metrics
            """
            combined_data: Dict[str, Any] = {}

            for metric_config in metric_configs:
                process_results_fn = metric_config.get("process_results")
                metric_name = metric_config.get("name", "unknown")

                if process_results_fn:
                    try:
                        metric_data = process_results_fn(doc, results)
                        if isinstance(metric_data, dict):
                            combined_data.update(metric_data)
                            logger.debug(
                                "Successfully processed results for metric: %s",
                                metric_name,
                            )
                    except RuntimeError as e:
                        logger.error(
                            "Error calling process_results for metric %s: %s",
                            metric_name,
                            e,
                        )
                else:
                    logger.debug("No process_results for metric: %s", metric_name)

            return combined_data

        def __repr__(self) -> str:
            metrics = [mc.get("name", "unknown") for mc in metric_configs]
            return f"<CombinedFunction(metrics={metrics})>"

    return CombinedFunction()


# Global storage for the current combined function (used by the wrapper)
CURRENT_COMBINED_FUNCTION: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None


def _current_combined_process_results(
    doc: Dict[str, Any], results: Any
) -> Dict[str, Any]:
    """Current combined process_results function.

    This is a wrapper that delegates to the currently registered combined function.
    The function is set dynamically when tasks are created.

    Args:
        doc: Document containing reference output
        results: Model predictions

    Returns:
        Dictionary with combined metric data from all metrics
    """
    global CURRENT_COMBINED_FUNCTION  # pylint: disable=global-variable-not-assigned

    if CURRENT_COMBINED_FUNCTION is None:
        raise RuntimeError(
            "No combined function registered. "
            "Make sure to set CURRENT_COMBINED_FUNCTION before use."
        )

    return CURRENT_COMBINED_FUNCTION(doc, results)
