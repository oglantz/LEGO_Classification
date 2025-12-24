"""HuggingFace dataset wrapper for LEGO parts."""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from collections import Counter
import torchvision.transforms as transforms
from datasets import load_dataset
import io


class LEGODataset(Dataset):
    """
    Dataset wrapper for LEGO parts from HuggingFace.
    """
    
    def __init__(
        self,
        dataset_name: str = "pvrancx/legobricks",
        split: str = "train",
        image_size: int = 224,
        class_ids: Optional[List[int]] = None,
        top_n_classes: Optional[int] = None,
        augment: bool = False,
        augment_config: Optional[Dict] = None,
    ):
        """
        Initialize LEGO dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split ("train", "val", "test")
            image_size: Target image size
            class_ids: Specific class IDs to include (None = all)
            top_n_classes: Use top N most common classes (None = all)
            augment: Whether to apply augmentations
            augment_config: Augmentation configuration
        """
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        
        # Load dataset
        print(f"Loading dataset {dataset_name} (split: {split})...")
        try:
            self.dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            print("Attempting to load with alternative configuration...")
            try:
                # Try loading without split specification
                full_dataset = load_dataset(dataset_name)
                if split in full_dataset:
                    self.dataset = full_dataset[split]
                else:
                    # Use train split as fallback
                    available_splits = list(full_dataset.keys())
                    print(f"Available splits: {available_splits}")
                    self.dataset = full_dataset[available_splits[0]]
            except Exception as e2:
                raise RuntimeError(f"Failed to load dataset: {e2}")
        
        # Process labels
        self.label_to_idx, self.idx_to_label = self._process_labels(
            class_ids=class_ids,
            top_n_classes=top_n_classes
        )
        
        # Filter dataset to selected classes
        self.filtered_indices = self._filter_by_classes()
        
        # Setup transforms
        self.transform = self._get_transforms(augment_config)
        
        print(f"Dataset loaded: {len(self.filtered_indices)} samples, {len(self.label_to_idx)} classes")
    
    def _process_labels(
        self,
        class_ids: Optional[List[int]],
        top_n_classes: Optional[int],
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Process labels and create mapping.
        
        Args:
            class_ids: Specific class IDs to include
            top_n_classes: Use top N most common classes
            
        Returns:
            Tuple of (label_to_idx, idx_to_label) dictionaries
        """
        # Find the label column name
        self._label_field = None
        for field in ['label', 'class', 'part_id', 'part_num', 'id']:
            if field in self.dataset.column_names:
                self._label_field = field
                break
        
        if self._label_field is None:
            raise ValueError(f"Could not find label column. Available columns: {self.dataset.column_names}")
        
        # Extract all labels directly from the column (without loading images!)
        print(f"Extracting labels from column '{self._label_field}'...")
        all_labels = [str(label) for label in self.dataset[self._label_field]]
        
        # Count label frequencies
        label_counts = Counter(all_labels)
        print(f"Found {len(label_counts)} unique classes")
        
        # Select classes
        if class_ids is not None:
            selected_labels = [str(cid) for cid in class_ids if str(cid) in label_counts]
        elif top_n_classes is not None:
            # Get top N most common
            top_labels = label_counts.most_common(top_n_classes)
            selected_labels = [label for label, _ in top_labels]
        else:
            selected_labels = list(label_counts.keys())
        
        # Store all_labels for use in _filter_by_classes
        self._all_labels = all_labels
        
        # Create mapping to contiguous indices
        label_to_idx = {label: idx for idx, label in enumerate(sorted(selected_labels))}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        return label_to_idx, idx_to_label
    
    def _filter_by_classes(self) -> List[int]:
        """Filter dataset indices to only include selected classes."""
        # Use cached labels from _process_labels (avoids re-iterating dataset)
        filtered = [
            idx for idx, label in enumerate(self._all_labels)
            if label in self.label_to_idx
        ]
        
        # Clean up the cached labels to free memory
        del self._all_labels
        
        return filtered
    
    def _get_transforms(self, augment_config: Optional[Dict]) -> Callable:
        """
        Get image transforms.
        
        Args:
            augment_config: Augmentation configuration
            
        Returns:
            Transform function
        """
        if augment_config is None:
            augment_config = {}
        
        transform_list = []
        
        if self.augment:
            # Training augmentations
            transform_list.extend([
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=augment_config.get('brightness', 0.2),
                    contrast=augment_config.get('contrast', 0.2),
                    saturation=augment_config.get('saturation', 0.2),
                    hue=augment_config.get('hue', 0.1),
                ),
                transforms.RandomAffine(
                    degrees=augment_config.get('rotation', 15),
                    translate=(0.1, 0.1),
                ),
            ])
            
            # Add blur and noise if specified
            if augment_config.get('blur', False):
                transform_list.append(transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.3))
            
            # Random background (simulated with random crop variations)
        else:
            # Validation/test transforms
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Common transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.filtered_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get dataset item.
        
        Args:
            idx: Index in filtered dataset
            
        Returns:
            Tuple of (image_tensor, label_index)
        """
        dataset_idx = self.filtered_indices[idx]
        item = self.dataset[dataset_idx]
        
        # Load image
        image = None
        for field in ['image', 'img', 'picture']:
            if field in item:
                image = item[field]
                break
        
        if image is None:
            # Try to load from bytes or path
            for field in ['image_bytes', 'path', 'file_path']:
                if field in item:
                    if isinstance(item[field], bytes):
                        image = Image.open(io.BytesIO(item[field])).convert('RGB')
                    else:
                        image = Image.open(item[field]).convert('RGB')
                    break
        
        if image is None:
            raise ValueError(f"Could not find image in dataset item at index {dataset_idx}")
        
        # Convert PIL to RGB if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(np.array(image)).convert('RGB')
        else:
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Get label using cached field name
        label = str(item[self._label_field])
        
        if label not in self.label_to_idx:
            raise ValueError(f"Invalid label '{label}' for item at index {dataset_idx}")
        
        label_idx = self.label_to_idx[label]
        
        return image_tensor, label_idx
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.label_to_idx)
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        return self.idx_to_label.get(idx, f"unknown_{idx}")


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
    train_dataset = LEGODataset(
        dataset_name=dataset_name,
        split="train",
        image_size=image_size,
        top_n_classes=top_n_classes,
        augment=True,
        augment_config=augment_config,
    )
    
    # Try to create val and test datasets
    val_dataset = None
    test_dataset = None
    
    try:
        val_dataset = LEGODataset(
            dataset_name=dataset_name,
            split="validation",
            image_size=image_size,
            class_ids=list(train_dataset.label_to_idx.keys()),
            augment=False,
        )
    except:
        try:
            val_dataset = LEGODataset(
                dataset_name=dataset_name,
                split="val",
                image_size=image_size,
                class_ids=list(train_dataset.label_to_idx.keys()),
                augment=False,
            )
        except:
            print("Validation split not available, using train split for validation")
            val_dataset = None
    
    try:
        test_dataset = LEGODataset(
            dataset_name=dataset_name,
            split="test",
            image_size=image_size,
            class_ids=list(train_dataset.label_to_idx.keys()),
            augment=False,
        )
    except:
        print("Test split not available")
        test_dataset = None
    
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

