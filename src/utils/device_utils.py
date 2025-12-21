"""Device utilities for GPU/CPU detection and management."""

import torch
import random
import numpy as np


def get_device(use_cuda: bool = True):
	"""
	Get the appropriate device (CUDA if available, else CPU).
	"""
	if use_cuda and torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")
	else:
		device = torch.device("cpu")
		if use_cuda:
			print("CUDA not available, falling back to CPU")
		else:
			print("Using CPU")
	return device


def set_seed(seed: int = 42):
	"""
	Set random seeds for reproducibility.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


