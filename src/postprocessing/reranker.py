"""Geometric re-ranking for LEGO part predictions."""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from scipy import ndimage


class GeometricReranker:
    """
    Re-rank predictions using geometric heuristics.
    """
    
    def __init__(
        self,
        aspect_ratio_tolerance: float = 0.3,
        area_tolerance: float = 0.5,
        stud_count_weight: float = 0.2,
    ):
        """
        Initialize geometric re-ranker.
        
        Args:
            aspect_ratio_tolerance: Tolerance for aspect ratio matching (0-1)
            area_tolerance: Tolerance for area matching (0-1)
            stud_count_weight: Weight for stud count heuristic (0-1)
        """
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.area_tolerance = area_tolerance
        self.stud_count_weight = stud_count_weight
    
    def rerank(
        self,
        predictions: List[Dict],
        masks: List[np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
        expected_aspect_ratios: Optional[Dict[int, float]] = None,
        expected_areas: Optional[Dict[int, float]] = None,
    ) -> List[Dict]:
        """
        Re-rank predictions based on geometric heuristics.
        
        Args:
            predictions: List of prediction dicts with 'predicted_ids', 'confidences', 'top_k_predictions'
            masks: List of binary masks
            boxes: List of bounding boxes
            expected_aspect_ratios: Dict mapping class_id to expected aspect ratio (optional)
            expected_areas: Dict mapping class_id to expected area range (optional)
            
        Returns:
            List of re-ranked prediction dicts
        """
        reranked = []
        
        for pred, mask, box in zip(predictions, masks, boxes):
            # Compute geometric features
            aspect_ratio = self._compute_aspect_ratio(box)
            area = self._compute_area(mask)
            stud_count_proxy = self._estimate_stud_count(mask)
            
            # Get top predictions
            top_k = pred.get('top_k_predictions', [])
            if len(top_k) == 0:
                reranked.append(pred)
                continue
            
            # Score each prediction based on geometric consistency
            scored_predictions = []
            for class_id, confidence in top_k:
                score = confidence  # Start with original confidence
                
                # Aspect ratio matching
                if expected_aspect_ratios and class_id in expected_aspect_ratios:
                    expected_ar = expected_aspect_ratios[class_id]
                    ar_diff = abs(aspect_ratio - expected_ar) / max(aspect_ratio, expected_ar)
                    if ar_diff <= self.aspect_ratio_tolerance:
                        score += 0.1 * (1 - ar_diff / self.aspect_ratio_tolerance)
                    else:
                        score -= 0.1 * min(ar_diff / self.aspect_ratio_tolerance, 1.0)
                
                # Area matching
                if expected_areas and class_id in expected_areas:
                    expected_area = expected_areas[class_id]
                    if isinstance(expected_area, (list, tuple)):
                        min_area, max_area = expected_area
                        if min_area <= area <= max_area:
                            score += 0.1
                        else:
                            # Penalize if outside range
                            if area < min_area:
                                score -= 0.1 * (1 - area / min_area)
                            else:
                                score -= 0.1 * min((area - max_area) / max_area, 1.0)
                    else:
                        area_diff = abs(area - expected_area) / max(area, expected_area)
                        if area_diff <= self.area_tolerance:
                            score += 0.1 * (1 - area_diff / self.area_tolerance)
                        else:
                            score -= 0.1 * min(area_diff / self.area_tolerance, 1.0)
                
                # Stud count proxy (simple heuristic)
                # This is a placeholder - in practice, you'd need a database of expected stud counts
                # For now, we use edge density as a proxy
                if stud_count_proxy > 0:
                    # Higher edge density might indicate more studs
                    # This is a very rough heuristic
                    score += self.stud_count_weight * min(stud_count_proxy / 10.0, 1.0)
                
                # Ensure score is in valid range
                score = max(0.0, min(1.0, score))
                
                scored_predictions.append((class_id, confidence, score))
            
            # Sort by new score
            scored_predictions.sort(key=lambda x: x[2], reverse=True)
            
            # Update prediction dict
            new_pred = pred.copy()
            new_pred['predicted_ids'] = [int(p[0]) for p in scored_predictions]
            new_pred['confidences'] = [float(p[2]) for p in scored_predictions]
            new_pred['top_k_predictions'] = [(int(p[0]), float(p[2])) for p in scored_predictions]
            new_pred['original_confidences'] = [float(p[1]) for p in scored_predictions]
            new_pred['geometric_features'] = {
                'aspect_ratio': aspect_ratio,
                'area': area,
                'stud_count_proxy': stud_count_proxy,
            }
            
            reranked.append(new_pred)
        
        return reranked
    
    def _compute_aspect_ratio(self, box: Tuple[int, int, int, int]) -> float:
        """
        Compute aspect ratio (width/height) of bounding box.
        
        Args:
            box: Bounding box as (x_min, y_min, x_max, y_max)
            
        Returns:
            Aspect ratio
        """
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        if height == 0:
            return float('inf')
        
        return width / height
    
    def _compute_area(self, mask: np.ndarray) -> float:
        """
        Compute area of mask in pixels.
        
        Args:
            mask: Binary mask
            
        Returns:
            Area in pixels
        """
        mask_binary = (mask > 128).astype(np.uint8)
        return float(np.sum(mask_binary))
    
    def _estimate_stud_count(self, mask: np.ndarray) -> float:
        """
        Estimate stud count using edge detection as a proxy.
        
        This is a rough heuristic - actual stud counting would require
        more sophisticated computer vision techniques.
        
        Args:
            mask: Binary mask
            
        Returns:
            Stud count proxy (normalized 0-1)
        """
        mask_binary = (mask > 128).astype(np.uint8)
        
        # Apply edge detection
        edges = cv2.Canny(mask_binary, 50, 150)
        
        # Count edge pixels (proxy for complexity/stud count)
        edge_density = np.sum(edges > 0) / max(np.sum(mask_binary > 0), 1)
        
        # Normalize to 0-1 range (rough heuristic)
        # Higher edge density might indicate more studs/details
        return min(edge_density * 10, 1.0)
    
    def compute_geometric_features(
        self,
        masks: List[np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Dict]:
        """
        Compute geometric features for all instances.
        
        Args:
            masks: List of masks
            boxes: List of boxes
            
        Returns:
            List of feature dictionaries
        """
        features = []
        
        for mask, box in zip(masks, boxes):
            features.append({
                'aspect_ratio': self._compute_aspect_ratio(box),
                'area': self._compute_area(mask),
                'stud_count_proxy': self._estimate_stud_count(mask),
            })
        
        return features

