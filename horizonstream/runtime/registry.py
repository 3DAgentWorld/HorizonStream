import torch.nn as nn


class SimpleRegistry:
    def __init__(self):
        self._items = {}

    def register_module(self, name=None):
        def decorator(obj):
            key = name or obj.__name__
            self._items[key] = obj
            return obj

        return decorator

    def build(self, cfg):
        if cfg is None:
            raise ValueError("Registry build requires a config or type")
        if isinstance(cfg, str):
            type_name = cfg
            kwargs = {}
        elif isinstance(cfg, dict):
            cfg = dict(cfg)
            type_name = cfg.pop("type", None)
            if type_name is None:
                raise KeyError("Registry config missing `type`")
            kwargs = cfg
        else:
            raise TypeError(f"Unsupported registry config type: {type(cfg)!r}")
        if type_name not in self._items:
            raise KeyError(f"Unknown registry type: {type_name}")
        return self._items[type_name](**kwargs)


REGRESSORS = SimpleRegistry()


def get_function(name):
    if not isinstance(name, str):
        raise TypeError(f"Activation name must be a string, got: {type(name)!r}")
    if not hasattr(nn, name):
        raise KeyError(f"Unknown torch.nn activation: {name}")
    return getattr(nn, name)()

