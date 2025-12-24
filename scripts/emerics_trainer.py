import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from collections import Counter
import torchvision.transforms as transforms
from datasets import load_dataset
import io

def create_dataloaders(
    dataset_name: str = "pvrancx/legobricks",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    top_n_classes: Optional[int] = None,
    augment_config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, val, and test dataloaders.
    
    Args:
        dataset_name: HuggingFace dataset name
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        top_n_classes: Use top N most common classes
        augment_config: Augmentation configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    # train_dataset = LEGODataset(
    #     dataset_name=dataset_name,
    #     split="train",
    #     image_size=image_size,
    #     top_n_classes=top_n_classes,
    #     augment=True,
    #     augment_config=augment_config,
    # )
    
    # Try to create val and test datasets
    val_dataset = None
    test_dataset = None
    
    # try:
    #     val_dataset = LEGODataset(
    #         dataset_name=dataset_name,
    #         split="validation",
    #         image_size=image_size,
    #         class_ids=list(train_dataset.label_to_idx.keys()),
    #         augment=False,
    #     )
    # except:
    #     try:
    #         val_dataset = LEGODataset(
    #             dataset_name=dataset_name,
    #             split="val",
    #             image_size=image_size,
    #             class_ids=list(train_dataset.label_to_idx.keys()),
    #             augment=False,
    #         )
    #     except:
    #         print("Validation split not available, using train split for validation")
    #         val_dataset = None
    
    # try:
    #     test_dataset = LEGODataset(
    #         dataset_name=dataset_name,
    #         split="test",
    #         image_size=image_size,
    #         class_ids=list(train_dataset.label_to_idx.keys()),
    #         augment=False,
    #     )
    # except:
    #     print("Test split not available")
    #     test_dataset = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader

dataset_name = "pvrancx/legobricks"
image_size = 224
num_classes = 3

train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name="pvrancx/legobricks",
            image_size=224,
            batch_size=10,
            num_workers=1,
            top_n_classes=3
        )

