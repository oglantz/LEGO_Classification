"""Segment Anything (SAM) wrapper for instance segmentation."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch
from PIL import Image

from ..utils.device_utils import get_device


class SAMWrapper:
	"""
	Wrapper around Meta's Segment Anything (SAM) automatic mask generator.
	"""

	def __init__(
		self,
		checkpoint_path: str,
		model_type: str = "vit_b",
		device: Optional[torch.device] = None,
		points_per_side: int = 32,
		pred_iou_thresh: float = 0.88,
		stability_score_thresh: float = 0.95,
		min_mask_area: int = 100,
	):
		"""
		Args:
			checkpoint_path: Path to SAM .pth weights (e.g., sam_vit_b_01ec64.pth)
			model_type: SAM model type key (e.g., "vit_b", "vit_l", "vit_h")
			device: torch device; auto-detected if None
			points_per_side: grid density for mask proposals
			pred_iou_thresh: filter low-IoU masks
			stability_score_thresh: filter unstable masks
			min_mask_area: remove tiny masks (in pixels)
		"""
		if device is None:
			self.device = get_device()
		else:
			self.device = device

		ckpt = Path(checkpoint_path)
		if not ckpt.exists():
			raise FileNotFoundError(
				f"SAM checkpoint not found at {ckpt}. "
				"Download sam_vit_b_01ec64.pth and update configs/default.yaml."
			)

		try:
			from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
		except Exception as e:
			raise RuntimeError(
				"segment-anything is not installed. Install with:\n"
				'  pip install "git+https://github.com/facebookresearch/segment-anything.git"\n'
				f"Original import error: {e}"
			)

		sam = sam_model_registry[model_type](checkpoint=str(ckpt))
		sam.to(device=self.device)

		self.mask_generator = SamAutomaticMaskGenerator(
			model=sam,
			points_per_side=points_per_side,
			pred_iou_thresh=pred_iou_thresh,
			stability_score_thresh=stability_score_thresh,
			min_mask_region_area=min_mask_area,
		)

	def segment(
		self,
		image: np.ndarray,
		return_masks: bool = True,
		return_boxes: bool = True,
	) -> Dict[str, List]:
		"""
		Generate segmentation masks for an image.
		Returns dict with 'masks', 'boxes', 'scores'.
		"""
		# Normalize input to numpy RGB
		if isinstance(image, Image.Image):
			image_np = np.array(image.convert("RGB"))
		else:
			image_np = image

		masks_out: List[np.ndarray] = []
		boxes_out: List[Tuple[int, int, int, int]] = []
		scores_out: List[float] = []

		try:
			anns = self.mask_generator.generate(image_np)
			for ann in anns:
				if return_masks:
					mask = ann["segmentation"].astype(np.uint8) * 255
					masks_out.append(mask)
				if return_boxes:
					x, y, w, h = ann["bbox"]
					boxes_out.append((int(x), int(y), int(x + w), int(y + h)))
				scores_out.append(float(ann.get("predicted_iou", 0.0)))
		except Exception:
			# Very simple fallback to keep pipeline alive
			import cv2
			gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
			_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			for c in contours:
				area = cv2.contourArea(c)
				if area < 100:
					continue
				mask = np.zeros(gray.shape, dtype=np.uint8)
				cv2.fillPoly(mask, [c], 255)
				masks_out.append(mask)
				x, y, w, h = cv2.boundingRect(c)
				boxes_out.append((x, y, x + w, y + h))
				scores_out.append(0.5)

		return {
			"masks": masks_out if return_masks else [],
			"boxes": boxes_out if return_boxes else [],
			"scores": scores_out,
		}


