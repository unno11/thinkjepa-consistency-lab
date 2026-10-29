"""Independent, inference-only evaluation utilities for ThinkJEPA.

Metric dependencies are imported lazily so protocol and configuration checks do
not require NumPy/PyTorch in lightweight environments.
"""

__all__ = ["compute_trajectory_metrics"]


def __getattr__(name):
    if name == "compute_trajectory_metrics":
        from .metrics import compute_trajectory_metrics

        return compute_trajectory_metrics
    raise AttributeError(name)
