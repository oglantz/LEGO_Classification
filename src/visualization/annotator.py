"""Visualization tools for annotated outputs."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional


class Annotator:
	"""
	Annotate images with boxes and labels (mask overlay optional/minimal for MVP).
	"""

	def __init__(
		self,
		font_size: int = 16,
		mask_alpha: float = 0.4,
		show_boxes: bool = True,
		show_labels: bool = True,
		show_confidences: bool = True,
	):
		self.font_size = font_size
		self.mask_alpha = mask_alpha
		self.show_boxes = show_boxes
		self.show_labels = show_labels
		self.show_confidences = show_confidences
		try:
			self.font = ImageFont.truetype("arial.ttf", font_size)
		except:
			self.font = ImageFont.load_default()

	def annotate(
		self,
		image: np.ndarray,
		masks: List[np.ndarray],
		boxes: List[Tuple[int, int, int, int]],
		predictions: List[Dict],
		class_names: Optional[Dict[int, str]] = None,
	) -> np.ndarray:
		pil = Image.fromarray(image.astype(np.uint8))
		overlay = pil.copy()
		draw = ImageDraw.Draw(overlay)

		colors = self._colors(len(boxes))

		for i, (box, pred) in enumerate(zip(boxes, predictions)):
			color = tuple(int(c * 255) for c in colors[i])

			# draw box
			if self.show_boxes:
				draw.rectangle(box, outline=color, width=2)

			# draw label
			if self.show_labels and pred.get("predicted_ids"):
				class_id = pred["predicted_ids"][0]
				conf = pred["confidences"][0] if pred.get("confidences") else 0.0
				name = class_names.get(class_id, f"Part_{class_id}") if class_names else f"Part_{class_id}"
				text = f"{name} ({conf:.2f})" if self.show_confidences else name
				x1, y1, x2, y2 = box
				tx, ty = x1, max(0, y1 - 18)
				bbox = draw.textbbox((tx, ty), text, font=self.font)
				tw = bbox[2] - bbox[0]
				th = bbox[3] - bbox[1]
				draw.rectangle([tx, ty, tx + tw + 6, ty + th], outline=color, width=2)
				draw.text((tx + 3, ty + 2), text, fill=(255, 255, 255), font=self.font)

		# Simple alpha blend of overlays (masks can be incorporated later)
		alpha = self.mask_alpha if masks else 0.0
		return np.array(Image.blend(pil, overlay, alpha))

	def _colors(self, n: int):
		# Generate distinct colors in HSV wheel
		if n <= 0:
			return []
		return [self._hsv_to_rgb(i / max(n, 1), 0.7, 0.9) for i in range(n)]

	def _hsv_to_rgb(self, h, s, v):
		i = int(h * 6)
		f = h * 6 - i
		p = v * (1 - s)
		q = v * (1 - f * s)
		t = v * (1 - (1 - f) * s)
		i = i % 6
		if i == 0:
			r, g, b = v, t, p
		elif i == 1:
			r, g, b = q, v, p
		elif i == 2:
			r, g, b = p, v, t
		elif i == 3:
			r, g, b = p, q, v
		elif i == 4:
			r, g, b = t, p, v
		else:
			r, g, b = v, p, q
		return (r, g, b)


