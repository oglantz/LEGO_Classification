"""Segmentation module for instance discovery."""

from .fastsam_wrapper import FastSAMWrapper
from .mask_processor import MaskProcessor

__all__ = ["FastSAMWrapper", "MaskProcessor"]

