"""ResNet50 classifier for LEGO part identification."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..utils.device_utils import get_device


class LEGOClassifier(nn.Module):
    """
    ResNet50-based classifier for LEGO part identification.
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """
        Initialize LEGO classifier.
        
        Args:
            num_classes: Number of LEGO part classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate for final layer
        """
        super(LEGOClassifier, self).__init__()
        
        # Load ResNet50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits tensor (B, num_classes)
        """
        return self.backbone(x)
    
    def predict(
        self,
        image: torch.Tensor,
        top_k: int = 5,
        confidence_threshold: float = 0.3,
        return_probs: bool = True,
    ) -> Dict[str, List]:
        """
        Predict class for input image.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence threshold
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary with:
                - 'predicted_ids': List of predicted class IDs (top-k)
                - 'confidences': List of confidence scores
                - 'top_k_predictions': List of (class_id, confidence) tuples
        """
        self.eval()
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(image)
            
            if return_probs:
                probs = torch.softmax(logits, dim=1)
            else:
                probs = logits
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes), dim=1)
            
            # Filter by confidence threshold
            top_probs = top_probs[0].cpu().numpy()
            top_indices = top_indices[0].cpu().numpy()
            
            # Filter by threshold
            valid_mask = top_probs >= confidence_threshold
            filtered_probs = top_probs[valid_mask]
            filtered_indices = top_indices[valid_mask]
            
            # If nothing passes threshold, return top-1 anyway
            if len(filtered_indices) == 0:
                filtered_probs = top_probs[:1]
                filtered_indices = top_indices[:1]
            
            # Sort by confidence (descending)
            sort_idx = np.argsort(filtered_probs)[::-1]
            filtered_probs = filtered_probs[sort_idx]
            filtered_indices = filtered_indices[sort_idx]
            
            # Create top-k list
            top_k_list = [
                (int(idx), float(prob))
                for idx, prob in zip(filtered_indices, filtered_probs)
            ]
        
        return {
            'predicted_ids': [int(idx) for idx in filtered_indices],
            'confidences': [float(prob) for prob in filtered_probs],
            'top_k_predictions': top_k_list,
        }
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        num_classes: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> 'LEGOClassifier':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            num_classes: Number of classes (if not in checkpoint)
            device: Device to load model on
            
        Returns:
            Loaded LEGOClassifier instance
        """
        if device is None:
            device = get_device()
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get num_classes from checkpoint or use provided
        if num_classes is None:
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
            elif 'model_state_dict' in checkpoint:
                # Try to infer from state dict
                state_dict = checkpoint['model_state_dict']
                if 'backbone.fc.1.weight' in state_dict:
                    num_classes = state_dict['backbone.fc.1.weight'].shape[0]
                else:
                    raise ValueError("Cannot determine num_classes from checkpoint")
            else:
                raise ValueError("num_classes must be provided or in checkpoint")
        
        # Create model
        model = cls(num_classes=num_classes, pretrained=False)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model


def create_dummy_classifier(num_classes: int = 1000) -> LEGOClassifier:
    """
    Create a dummy classifier that returns random predictions.
    Useful for testing the pipeline without trained weights.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        LEGOClassifier instance (not trained)
    """
    model = LEGOClassifier(num_classes=num_classes, pretrained=True)
    # Keep ImageNet weights but final layer will be random
    # This allows the pipeline to run end-to-end
    return model

