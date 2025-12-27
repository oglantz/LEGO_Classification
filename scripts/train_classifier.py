#!/usr/bin/env python3
"""Training script for LEGO classifier."""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classification import LEGOClassifier, Trainer
from src.classification.dataset import create_dataloaders, create_dataloaders_from_disk
from src.utils import load_config, get_config_value, get_device, set_seed


def train_classifier(
    config_path: str = "configs/default.yaml",
    resume_from: str = None,
):
    """
    Train LEGO classifier.
    
    Args:
        config_path: Path to config file
        resume_from: Path to checkpoint to resume from
    """
    print("=" * 60)
    print("LEGO Classifier Training")
    print("=" * 60)
    
    # Load configuration
    print(f"\nLoading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Load augmentation config if available
    aug_config_path = Path("configs/augmentation.yaml")
    aug_config = None
    if aug_config_path.exists():
        aug_config = load_config(str(aug_config_path))
    
    # Setup device and seed
    use_cuda = get_config_value(config, "device.use_cuda", True)
    device = get_device(use_cuda=use_cuda)
    seed = get_config_value(config, "device.seed", 42)
    set_seed(seed)
    
    # Get data configuration
    data_config = get_config_value(config, "data", {})
    dataset_name = get_config_value(data_config, "dataset_name", "pvrancx/legobricks")
    image_size = get_config_value(data_config, "image_size", 224)
    batch_size = get_config_value(data_config, "batch_size", 32)
    splits_dir = get_config_value(data_config, "splits_dir", None)
    num_workers = get_config_value(data_config, "num_workers", 8)
    prefetch_factor = get_config_value(data_config, "prefetch_factor", 4)
    use_gpu_aug = get_config_value(data_config, "use_gpu_aug", False)
    
    # Get classification configuration
    cls_config = get_config_value(config, "classification", {})
    num_classes = get_config_value(cls_config, "num_classes", 1000)
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    print(f"Dataset: {dataset_name}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    
    try:
        # Prefer custom splits if provided
        if splits_dir is not None and Path(splits_dir).exists():
            print(f"Loading custom splits from: {splits_dir}")
            train_loader, val_loader, test_loader = create_dataloaders_from_disk(
                dataset_path=splits_dir,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                use_gpu_aug=use_gpu_aug,
                augment_config=aug_config,
            )
        else:
            train_loader, val_loader, test_loader = create_dataloaders(
                dataset_name=dataset_name,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                use_gpu_aug=use_gpu_aug,
                top_n_classes=num_classes,
                augment_config=aug_config,
            )
        
        # Update num_classes from actual dataset
        actual_num_classes = train_loader.dataset.get_num_classes()
        if actual_num_classes != num_classes:
            print(f"Warning: Dataset has {actual_num_classes} classes, but config specifies {num_classes}")
            print(f"Using {actual_num_classes} classes from dataset")
            num_classes = actual_num_classes
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create model
    print(f"\nCreating model...")
    model = LEGOClassifier(
        num_classes=num_classes,
        pretrained=True,  # Use ImageNet pretrained weights
    )
    print(f"Model created with {num_classes} classes")
    
    # Get training configuration
    train_config = get_config_value(config, "training", {})
    learning_rate = get_config_value(train_config, "learning_rate", 0.001)
    weight_decay = get_config_value(train_config, "weight_decay", 0.0001)
    mixed_precision = get_config_value(train_config, "mixed_precision", True)
    grad_accum_steps = get_config_value(train_config, "grad_accum_steps", 1)
    checkpoint_dir = get_config_value(train_config, "checkpoint_dir", "models/classification")
    log_dir = get_config_value(train_config, "log_dir", "logs")
    early_stopping_patience = get_config_value(train_config, "early_stopping_patience", 10)
    num_epochs = get_config_value(train_config, "epochs", 50)
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mixed_precision=mixed_precision,
        grad_accum_steps=grad_accum_steps,
        use_gpu_aug=use_gpu_aug,
        image_size=image_size,
        augment_config=aug_config,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        early_stopping_patience=early_stopping_patience,
    )
    
    # Train
    print(f"\nStarting training for {num_epochs} epochs...")
    trainer.train(num_epochs=num_epochs, resume_from=resume_from)

    # Final test evaluation (optional, only if test split is available)
    if 'test_loader' in locals() and test_loader is not None:
        print("\nEvaluating on test set...")
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total = 0
        correct = 0
        running_loss = 0.0
        top5_correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)

                # Normalize on GPU if we skipped CPU normalization
                if use_gpu_aug:
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    images = (images - mean) / std

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Top-1 accuracy
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # Top-5 accuracy
                top5_indices = torch.topk(outputs, 5, dim=1).indices
                top5_correct += top5_indices.eq(labels.view(-1, 1)).any(dim=1).sum().item()

        test_loss = running_loss / len(test_loader)
        test_acc = 100.0 * correct / total if total > 0 else 0.0
        test_top5 = 100.0 * top5_correct / total if total > 0 else 0.0
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Top-5: {test_top5:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best checkpoint saved to: {checkpoint_dir}/best_checkpoint.pt")
    print(f"Last checkpoint saved to: {checkpoint_dir}/last_checkpoint.pt")
    print(f"TensorBoard logs saved to: {log_dir}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {log_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train LEGO piece classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Validate config
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Train
    try:
        train_classifier(
            config_path=args.config,
            resume_from=args.resume,
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

