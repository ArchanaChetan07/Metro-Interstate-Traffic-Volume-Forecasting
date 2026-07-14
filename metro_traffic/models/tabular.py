"""Backward-compatible re-exports for tabular models."""

from metro_traffic.models.gbr import eval_model, train_gbr
from metro_traffic.models.linear import train_linear

__all__ = ["train_linear", "train_gbr", "eval_model"]
