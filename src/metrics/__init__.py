"""Metrics package initialization (auto-discovery)."""

import importlib
import pkgutil

__all__ = []

for module_info in pkgutil.iter_modules(__path__):
    if module_info.name.endswith("_metric"):
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        __all__.append(module_info.name)
