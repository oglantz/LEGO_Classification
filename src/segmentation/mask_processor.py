"""Mask post-processing for cleaning and filtering."""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from scipy import ndimage


class MaskProcessor:
    """
    Post-process segmentation masks to remove noise, merge overlaps, and filter.
    """
    
    def __init__(
        self,
        min_mask_area: int = 100,
        overlap_threshold: float = 0.7,
        min_aspect_ratio: float = 0.1,
        max_aspect_ratio: float = 10.0,
        morph_kernel_size: int = 3,
    ):
        """
        Initialize mask processor.
        
        Args:
            min_mask_area: Minimum mask area in pixels (masks smaller are removed)
            overlap_threshold: IoU threshold for merging overlapping masks
            min_aspect_ratio: Minimum bounding box aspect ratio (width/height)
            max_aspect_ratio: Maximum bounding box aspect ratio (width/height)
            morph_kernel_size: Kernel size for morphological operations
        """
        self.min_mask_area = min_mask_area
        self.overlap_threshold = overlap_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.morph_kernel_size = morph_kernel_size
    
    def process_masks(
        self,
        masks: List[np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
    ) -> Dict[str, List]:
        """
        Process masks: clean, filter, and merge overlaps.
        
        Args:
            masks: List of binary masks (H, W) as numpy arrays
            boxes: List of bounding boxes as (x_min, y_min, x_max, y_max)
            scores: List of confidence scores
            
        Returns:
            Dictionary with processed 'masks', 'boxes', 'scores'
        """
        if len(masks) == 0:
            return {'masks': [], 'boxes': [], 'scores': []}
        
        # Step 1: Clean individual masks
        cleaned_masks = []
        cleaned_boxes = []
        cleaned_scores = []
        
        for mask, box, score in zip(masks, boxes, scores):
            # Clean mask
            mask_cleaned = self._clean_mask(mask)
            
            # Check area
            area = np.sum(mask_cleaned > 0)
            if area < self.min_mask_area:
                continue
            
            # Check aspect ratio
            if not self._check_aspect_ratio(box):
                continue
            
            # Update box based on cleaned mask
            box_updated = self._mask_to_bbox(mask_cleaned)
            if box_updated is None:
                continue
            
            cleaned_masks.append(mask_cleaned)
            cleaned_boxes.append(box_updated)
            cleaned_scores.append(score)
        
        # Step 2: Remove overlapping masks
        filtered_masks, filtered_boxes, filtered_scores = self._remove_overlaps(
            cleaned_masks, cleaned_boxes, cleaned_scores
        )
        
        return {
            'masks': filtered_masks,
            'boxes': filtered_boxes,
            'scores': filtered_scores,
        }
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean a single mask using morphological operations.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Cleaned binary mask
        """
        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask = (mask > 128).astype(np.uint8) * 255
        
        # Morphological opening (removes small noise)
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Morphological closing (fills small holes)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _check_aspect_ratio(self, box: Tuple[int, int, int, int]) -> bool:
        """
        Check if bounding box aspect ratio is within acceptable range.
        
        Args:
            box: Bounding box as (x_min, y_min, x_max, y_max)
            
        Returns:
            True if aspect ratio is acceptable
        """
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        if height == 0:
            return False
        
        aspect_ratio = width / height
        return self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute bounding box from mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Bounding box as (x_min, y_min, x_max, y_max) or None if mask is empty
        """
        mask_binary = (mask > 128).astype(np.uint8)
        
        # Find non-zero pixels
        coords = np.column_stack(np.where(mask_binary > 0))
        
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two masks.
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            IoU value (0-1)
        """
        mask1_binary = (mask1 > 128).astype(np.uint8)
        mask2_binary = (mask2 > 128).astype(np.uint8)
        
        intersection = np.logical_and(mask1_binary, mask2_binary).sum()
        union = np.logical_or(mask1_binary, mask2_binary).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def _remove_overlaps(
        self,
        masks: List[np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[float]]:
        """
        Remove overlapping masks, keeping the one with higher score.
        
        Args:
            masks: List of masks
            boxes: List of boxes
            scores: List of scores
            
        Returns:
            Filtered lists of masks, boxes, scores
        """
        if len(masks) == 0:
            return [], [], []
        
        # Sort by score (descending)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        filtered_masks = []
        filtered_boxes = []
        filtered_scores = []
        filtered_indices = []
        
        for idx in indices:
            mask = masks[idx]
            box = boxes[idx]
            score = scores[idx]
            
            # Check overlap with already filtered masks
            has_significant_overlap = False
            for fidx in filtered_indices:
                iou = self._compute_iou(mask, masks[fidx])
                if iou > self.overlap_threshold:
                    has_significant_overlap = True
                    break
            
            if not has_significant_overlap:
                filtered_masks.append(mask)
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_indices.append(idx)
        
        return filtered_masks, filtered_boxes, filtered_scores

