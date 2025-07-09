# Helpers/__init__.py

import pkgutil
import importlib
import inspect
from pathlib import Path

__all__ = []

# Path to this package
_pkg_dir = Path(__file__).parent

for finder, module_name, ispkg in pkgutil.iter_modules([str(_pkg_dir)]):
    # skip private modules
    if module_name.startswith('_'):
        continue

    # import the module under Helpers
    full_module_name = f"{__name__}.{module_name}"
    module = importlib.import_module(full_module_name)

    # find every class defined in that module
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # only include classes actually defined in that module
        if obj.__module__ == full_module_name:
            globals()[name] = obj
            __all__.append(name)
