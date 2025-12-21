"""Image I/O and transformation utilities."""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Union
import torch
from torchvision import transforms


def load_image(image_path: str) -> np.ndarray:
	"""
	Load an image from file path.
	
	Returns:
		RGB numpy array (H, W, 3)
	"""
	try:
		img = Image.open(image_path).convert("RGB")
		return np.array(img)
	except FileNotFoundError:
		raise FileNotFoundError(f"Image file not found: {image_path}")
	except Exception as e:
		raise ValueError(f"Failed to load image {image_path}: {str(e)}")


def save_image(image: Union[np.ndarray, Image.Image], output_path: str):
	"""Save an image to file."""
	if isinstance(image, np.ndarray):
		img = Image.fromarray(image.astype(np.uint8))
	else:
		img = image
	img.save(output_path)
	print(f"Saved image to {output_path}")


def apply_transforms(
	image: np.ndarray,
	size: Tuple[int, int] = (224, 224),
	normalize: bool = True,
	mean: Optional[Tuple[float, float, float]] = None,
	std: Optional[Tuple[float, float, float]] = None
) -> torch.Tensor:
	"""
	Apply standard image transforms for classification. Returns tensor (C, H, W).
	"""
	if mean is None:
		mean = (0.485, 0.456, 0.406)
	if std is None:
		std = (0.229, 0.224, 0.225)

	# Build transforms depending on input type
	is_pil = isinstance(image, Image.Image)

	transform_list = []
	if not is_pil:
		transform_list.append(transforms.ToPILImage())
	transform_list.extend([
		transforms.Resize(size),
		transforms.ToTensor(),
	])
	if normalize:
		transform_list.append(transforms.Normalize(mean=mean, std=std))
	transform = transforms.Compose(transform_list)

	return transform(image)


def extract_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
	"""Extract ROI from image using (x_min, y_min, x_max, y_max)."""
	x_min, y_min, x_max, y_max = bbox
	return image[y_min:y_max, x_min:x_max]


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Apply binary mask (0/255) to image.
	"""
	if mask.dtype != np.uint8:
		mask = (mask * 255).astype(np.uint8)
	if len(mask.shape) == 3:
		mask = mask[:, :, 0]
	masked = image.copy()
	mask_bool = mask > 128
	masked[~mask_bool] = 0
	return masked


