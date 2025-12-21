"""Classification module for per-instance identification."""

from .classifier import LEGOClassifier, create_dummy_classifier
from .dataset import LEGODataset
from .trainer import Trainer

__all__ = ["LEGOClassifier", "create_dummy_classifier", "LEGODataset", "Trainer"]

