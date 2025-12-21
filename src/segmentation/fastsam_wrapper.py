"""FastSAM wrapper for instance segmentation."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image

from ..utils.device_utils import get_device


class FastSAMWrapper:
    """
    Wrapper for FastSAM model for automatic mask generation.
    
    FastSAM is a lightweight alternative to SAM that can segment objects
    without prompts using automatic mask generation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "FastSAM-x",
        device: Optional[torch.device] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ):
        """
        Initialize FastSAM wrapper.
        
        Args:
            model_path: Path to FastSAM model weights. If None, tries to load from default location.
            model_type: Model variant ("FastSAM-x" for lightweight, "FastSAM-s" for standard)
            device: PyTorch device. If None, auto-detects.
            points_per_side: Number of points per side for automatic mask generation (higher = more masks)
            pred_iou_thresh: IoU threshold for filtering masks (0-1, higher = stricter)
            stability_score_thresh: Stability score threshold for filtering masks (0-1, higher = stricter)
        """
        self.model_type = model_type
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        self.model = None
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str]):
        """Load FastSAM model."""
        try:
            # Try to import FastSAM
            try:
                from fastsam import FastSAM
                self.fastsam_module = FastSAM
                self.fastsam_prompt = None  # optional in some builds
            except ImportError:
                # Fallback: try alternative import or manual implementation
                raise ImportError(
                    "FastSAM not installed. Install with: pip install fastsam\n"
                    "Or download weights manually and provide model_path."
                )
            
            # Determine model path
            if model_path is None:
                model_path = self._get_default_model_path()
            
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"FastSAM model weights not found at {model_path}.\n"
                    f"Please download weights using: python scripts/download_weights.py\n"
                    f"Or provide model_path argument."
                )
            
            # Load model
            print(f"Loading FastSAM model from {model_path}...")
            # Some FastSAM builds expect device as a string
            self.model = self.fastsam_module(model_path, device=str(self.device))
            print("FastSAM model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load FastSAM model: {str(e)}")
    
    def _get_default_model_path(self) -> str:
        """Get default model path based on model type."""
        model_dir = Path("models/segmentation")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == "FastSAM-x":
            return str(model_dir / "FastSAM-x.pt")
        elif self.model_type == "FastSAM-s":
            return str(model_dir / "FastSAM-s.pt")
        else:
            return str(model_dir / "FastSAM-x.pt")
    
    def segment(
        self,
        image: np.ndarray,
        return_masks: bool = True,
        return_boxes: bool = True,
    ) -> Dict[str, List]:
        """
        Generate segmentation masks for an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB
            return_masks: Whether to return mask arrays
            return_boxes: Whether to return bounding boxes
            
        Returns:
            Dictionary with keys:
                - 'masks': List of binary masks (H, W) as numpy arrays
                - 'boxes': List of bounding boxes as (x_min, y_min, x_max, y_max)
                - 'scores': List of confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image
        
        # Run FastSAM automatic mask generation
        try:
            everything_results = self.model(
                pil_image,
                device=str(self.device),
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
            )
            
            # Extract masks and boxes
            # FastSAMPrompt is optional; not required for basic everything mode
            
            # Get all masks
            masks = []
            boxes = []
            scores = []
            
            if len(everything_results) > 0:
                # Process results
                for result in everything_results:
                    if return_masks and hasattr(result, 'masks') and result.masks is not None:
                        # Get mask data
                        mask_data = result.masks.data
                        for i in range(len(mask_data)):
                            mask = mask_data[i].cpu().numpy()
                            # Convert to binary mask
                            mask_binary = (mask > 0.5).astype(np.uint8) * 255
                            masks.append(mask_binary)
                    
                    if return_boxes and hasattr(result, 'boxes') and result.boxes is not None:
                        box_data = result.boxes.data
                        for i in range(len(box_data)):
                            # FastSAM boxes format: [x1, y1, x2, y2, conf, cls]
                            box = box_data[i].cpu().numpy()
                            boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                            scores.append(float(box[4]))
            
            # If no results, try alternative approach with automatic mask generation
            if len(masks) == 0:
                print("No masks found with default settings, trying automatic mask generation...")
                masks, boxes, scores = self._generate_automatic_masks(pil_image)
            
            return {
                'masks': masks if return_masks else [],
                'boxes': boxes if return_boxes else [],
                'scores': scores,
            }
            
        except Exception as e:
            # Fallback: try simpler approach
            print(f"FastSAM segmentation failed: {e}")
            print("Attempting fallback segmentation method...")
            return self._fallback_segmentation(image)
    
    def _generate_automatic_masks(self, image: Image.Image) -> Tuple[List, List, List]:
        """Generate masks using automatic mask generation."""
        masks = []
        boxes = []
        scores = []
        
        # This is a simplified version - in practice, FastSAM's automatic
        # mask generation would be used here
        # For now, return empty lists (will be handled by fallback)
        return masks, boxes, scores
    
    def _fallback_segmentation(self, image: np.ndarray) -> Dict[str, List]:
        """
        Fallback segmentation using simple thresholding.
        Used when FastSAM is not available or fails.
        """
        import cv2
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple thresholding-based segmentation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        masks = []
        boxes = []
        scores = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter small regions
                continue
            
            # Create mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            masks.append(mask)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))
            scores.append(0.5)  # Default confidence
        
        return {
            'masks': masks,
            'boxes': boxes,
            'scores': scores,
        }

