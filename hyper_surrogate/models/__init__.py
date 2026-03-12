try:
    from hyper_surrogate.models.icnn import ICNN
    from hyper_surrogate.models.mlp import MLP

    __all__ = ["MLP", "ICNN"]
except ImportError:
    __all__ = []
