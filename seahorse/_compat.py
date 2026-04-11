"""Helpers for compatibility wrappers."""

from importlib import import_module
import sys
from types import ModuleType


def alias_module(alias_name: str, target_name: str) -> ModuleType:
    """Expose an existing module under a compatibility namespace."""
    module = import_module(target_name)
    sys.modules[alias_name] = module

    package_name, _, attr_name = alias_name.rpartition(".")
    if package_name:
        package = sys.modules.get(package_name)
        if package is not None:
            setattr(package, attr_name, module)

    return module
