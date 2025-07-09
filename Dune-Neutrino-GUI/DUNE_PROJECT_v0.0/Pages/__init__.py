import pkgutil, importlib
import tkinter as tk

__all__ = []

for finder, modname, ispkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{modname}")
    for attr in dir(module):
        obj = getattr(module, attr)
        # pick up any tk.Frame subclass defined in this module
        if isinstance(obj, type) and issubclass(obj, tk.Frame) and obj.__module__ == module.__name__:
            globals()[attr] = obj
            __all__.append(attr)
