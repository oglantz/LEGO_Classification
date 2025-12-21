#!/usr/bin/env python3
"""Main inference script for LEGO piece detection and classification."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation import FastSAMWrapper, MaskProcessor
from src.classification import LEGOClassifier, create_dummy_classifier
from src.postprocessing import GeometricReranker
from src.visualization import Annotator
from src.utils import (
    load_image, save_image, apply_transforms,
    get_device, load_config, get_config_value
)


def run_inference(
    image_path: str,
    config_path: str = "configs/default.yaml",
    output_dir: str = "outputs",
    checkpoint_path: str = None,
):
    """
    Run end-to-end inference pipeline.
    
    Args:
        image_path: Path to input image
        config_path: Path to config file
        output_dir: Output directory for results
        checkpoint_path: Path to classifier checkpoint (overrides config)
    """
    print("=" * 60)
    print("LEGO Piece Detection and Classification")
    print("=" * 60)
    
    # Load configuration
    print(f"\nLoading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Setup device
    use_cuda = get_config_value(config, "device.use_cuda", True)
    device = get_device(use_cuda=use_cuda)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load input image
    print(f"\nLoading image from {image_path}...")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")
    
    # ===== Stage A: Instance Segmentation =====
    print("\n" + "=" * 60)
    print("Stage A: Instance Segmentation")
    print("=" * 60)
    
    seg_config = get_config_value(config, "segmentation", {})
    
    # Initialize segmenter backend
    backend = get_config_value(seg_config, "backend", "fastsam").lower()
    print(f"Initializing segmenter backend: {backend}...")
    try:
        if backend == "sam":
            # Lazy import to avoid hard dependency when not used
            from src.segmentation.sam_wrapper import SAMWrapper
            segmenter = SAMWrapper(
                checkpoint_path=get_config_value(seg_config, "model_path"),
                model_type=get_config_value(seg_config, "model_type", "vit_b"),
                device=device,
                points_per_side=get_config_value(seg_config, "points_per_side", 32),
                pred_iou_thresh=get_config_value(seg_config, "pred_iou_thresh", 0.88),
                stability_score_thresh=get_config_value(seg_config, "stability_score_thresh", 0.95),
                min_mask_area=get_config_value(seg_config, "min_mask_area", 100),
            )
        else:
            # Default: FastSAM
            segmenter = FastSAMWrapper(
                model_path=get_config_value(seg_config, "model_path"),
                model_type=get_config_value(seg_config, "model_type", "FastSAM-x"),
                device=device,
                points_per_side=get_config_value(seg_config, "points_per_side", 32),
                pred_iou_thresh=get_config_value(seg_config, "pred_iou_thresh", 0.88),
                stability_score_thresh=get_config_value(seg_config, "stability_score_thresh", 0.95),
            )
    except Exception as e:
        print(f"Error initializing segmenter: {e}")
        print("Falling back to simple segmentation...")
        segmenter = None
    
    # Generate masks
    print("Generating segmentation masks...")
    if segmenter is not None:
        seg_results = segmenter.segment(image)
        masks = seg_results['masks']
        boxes = seg_results['boxes']
        scores = seg_results['scores']
    else:
        # Fallback: create dummy masks
        print("Using fallback segmentation (dummy masks)")
        masks = []
        boxes = []
        scores = []
    
    print(f"Generated {len(masks)} initial masks")
    
    # Post-process masks
    print("Post-processing masks...")
    processor = MaskProcessor(
        min_mask_area=get_config_value(seg_config, "min_mask_area", 100),
        overlap_threshold=get_config_value(seg_config, "overlap_threshold", 0.7),
    )
    
    processed = processor.process_masks(masks, boxes, scores)
    processed_masks = processed['masks']
    processed_boxes = processed['boxes']
    processed_scores = processed['scores']
    
    print(f"After post-processing: {len(processed_masks)} masks")
    
    if len(processed_masks) == 0:
        print("Warning: No valid masks found after post-processing!")
        print("The pipeline will still run but may produce empty results.")
    
    # ===== Stage B: Per-Instance Classification =====
    print("\n" + "=" * 60)
    print("Stage B: Per-Instance Classification")
    print("=" * 60)
    
    cls_config = get_config_value(config, "classification", {})
    num_classes = get_config_value(cls_config, "num_classes", 1000)
    top_k = get_config_value(cls_config, "top_k", 5)
    confidence_threshold = get_config_value(cls_config, "confidence_threshold", 0.3)
    
    # Load classifier
    print("Loading classifier...")
    checkpoint = checkpoint_path or get_config_value(cls_config, "checkpoint_path")
    
    if checkpoint and Path(checkpoint).exists():
        print(f"Loading from checkpoint: {checkpoint}")
        classifier = LEGOClassifier.load_from_checkpoint(
            checkpoint_path=checkpoint,
            num_classes=num_classes,
            device=device,
        )
    else:
        print("No checkpoint found, using dummy classifier (ImageNet-pretrained, random head)")
        print("Note: For best results, train the classifier first using train_classifier.py")
        classifier = create_dummy_classifier(num_classes=num_classes)
        classifier = classifier.to(device)
    
    # Classify each instance
    print(f"Classifying {len(processed_masks)} instances...")
    predictions = []
    
    data_config = get_config_value(config, "data", {})
    image_size = get_config_value(data_config, "image_size", 224)
    
    for idx, (mask, box) in enumerate(zip(processed_masks, processed_boxes)):
        # Extract ROI
        x_min, y_min, x_max, y_max = box
        roi = image[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            continue
        
        # Apply mask to ROI (keep 2D boolean mask to match HxW, applies to all 3 channels)
        mask_roi = mask[y_min:y_max, x_min:x_max]
        mask_bool = (mask_roi > 128)  # shape (H, W)
        masked_roi = roi.copy()
        masked_roi[~mask_bool] = 0
        
        # Transform for classification
        roi_tensor = apply_transforms(masked_roi, size=(image_size, image_size))
        roi_tensor = roi_tensor.to(device)
        
        # Predict
        pred = classifier.predict(
            roi_tensor,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )
        
        predictions.append(pred)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(processed_masks)} instances...")
    
    print(f"Completed classification of {len(predictions)} instances")
    
    # ===== Post-processing: Re-ranking =====
    print("\n" + "=" * 60)
    print("Post-processing: Geometric Re-ranking")
    print("=" * 60)
    
    rerank_config = get_config_value(config, "postprocessing", {})
    reranker = GeometricReranker(
        aspect_ratio_tolerance=get_config_value(rerank_config, "aspect_ratio_tolerance", 0.3),
        area_tolerance=get_config_value(rerank_config, "area_tolerance", 0.5),
        stud_count_weight=get_config_value(rerank_config, "stud_count_weight", 0.2),
    )
    
    reranked_predictions = reranker.rerank(
        predictions,
        processed_masks,
        processed_boxes,
    )
    
    # ===== Generate Outputs =====
    print("\n" + "=" * 60)
    print("Generating Outputs")
    print("=" * 60)
    
    # Create summary
    summary = {}
    for pred in reranked_predictions:
        if len(pred['predicted_ids']) > 0:
            class_id = pred['predicted_ids'][0]
            summary[class_id] = summary.get(class_id, 0) + 1
    
    # Prepare JSON output
    output_data = {
        'image_path': str(image_path),
        'num_instances': len(reranked_predictions),
        'instances': [],
        'summary': summary,
    }
    
    for idx, (mask, box, pred) in enumerate(zip(processed_masks, processed_boxes, reranked_predictions)):
        instance_data = {
            'instance_id': idx,
            'bbox': box,
            'predicted_id': pred['predicted_ids'][0] if len(pred['predicted_ids']) > 0 else None,
            'confidence': pred['confidences'][0] if len(pred['confidences']) > 0 else 0.0,
            'top_k_predictions': pred['top_k_predictions'],
        }
        
        if 'geometric_features' in pred:
            instance_data['geometric_features'] = pred['geometric_features']
        
        output_data['instances'].append(instance_data)
    
    # Save JSON
    json_path = output_path / f"{Path(image_path).stem}_results.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved JSON results to {json_path}")
    
    # Create annotated image
    print("Creating annotated image...")
    viz_config = get_config_value(config, "visualization", {})
    annotator = Annotator(
        font_size=get_config_value(viz_config, "font_size", 16),
        mask_alpha=get_config_value(viz_config, "mask_alpha", 0.4),
        show_boxes=get_config_value(viz_config, "show_boxes", True),
        show_labels=get_config_value(viz_config, "show_labels", True),
        show_confidences=get_config_value(viz_config, "show_confidences", True),
    )
    
    annotated_image = annotator.annotate(
        image,
        processed_masks,
        processed_boxes,
        reranked_predictions,
    )
    
    # Save annotated image
    annotated_path = output_path / f"{Path(image_path).stem}_annotated.png"
    save_image(annotated_image, str(annotated_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total instances detected: {len(reranked_predictions)}")
    print(f"Unique parts detected: {len(summary)}")
    print("\nPart counts:")
    for class_id, count in sorted(summary.items()):
        print(f"  Part {class_id}: {count}")
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LEGO piece detection and classification inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to classifier checkpoint (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run inference
    try:
        run_inference(
            image_path=args.image,
            config_path=args.config,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
        )
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

