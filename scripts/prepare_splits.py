#!/usr/bin/env python3
"""Prepare stratified train/val/test splits from a Hugging Face dataset.

Default behavior:
- Loads a single-split dataset (e.g., 'train') from Hugging Face (remote).
- Optionally restricts to top-N most frequent classes.
- Performs a stratified 80/10/10 split by the label column.
- Saves the resulting DatasetDict to disk for fast reuse.

You can later load these splits with datasets.load_from_disk(path) and point
your training pipeline at this directory.
"""

import argparse
from pathlib import Path
from typing import Optional, List, Tuple

from datasets import load_dataset, Dataset, DatasetDict


def detect_label_column(ds: Dataset) -> str:
    """Detect the label column name heuristically."""
    for field in ['label', 'class', 'part_id', 'part_num', 'id']:
        if field in ds.column_names:
            return field
    raise ValueError(f"Could not find label column. Available columns: {ds.column_names}")


def restrict_to_top_n_classes(
    ds: Dataset,
    label_col: str,
    top_n: int,
) -> Dataset:
    """Restrict a dataset to the top-N most frequent classes."""
    # Compute frequencies
    counts = {}
    for v in ds[label_col]:
        k = str(v)
        counts[k] = counts.get(k, 0) + 1
    # Determine top-N labels
    sorted_labels = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    keep_labels = set([k for k, _ in sorted_labels[:top_n]])
    # Filter
    return ds.filter(lambda x: str(x[label_col]) in keep_labels)


def stratified_splits(
    ds: Dataset,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> DatasetDict:
    """Create stratified train/val/test splits using two-stage splits."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    remaining_ratio = 1.0 - train_ratio
    first = ds.train_test_split(test_size=remaining_ratio, stratify_by_column=label_col, seed=seed)
    temp = first['test']
    # Split remaining into val/test maintaining the requested proportion
    test_share_of_temp = test_ratio / (val_ratio + test_ratio)
    second = temp.train_test_split(test_size=test_share_of_temp, stratify_by_column=label_col, seed=seed)
    return DatasetDict({
        'train': first['train'],
        'validation': second['train'],
        'test': second['test'],
    })


def random_splits(
    ds: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> DatasetDict:
    """Create random (non-stratified) train/val/test splits."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    remaining_ratio = 1.0 - train_ratio
    first = ds.train_test_split(test_size=remaining_ratio, seed=seed)
    temp = first['test']
    test_share_of_temp = test_ratio / (val_ratio + test_ratio)
    second = temp.train_test_split(test_size=test_share_of_temp, seed=seed)
    return DatasetDict({
        'train': first['train'],
        'validation': second['train'],
        'test': second['test'],
    })


def prepare_splits(
    dataset_name: str = "pvrancx/legobricks",
    split: str = "train",
    output_dir: str = "data/splits/legobricks",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    top_n_classes: Optional[int] = None,
    stratify: bool = True,
    seed: int = 42,
    shuffle_before: bool = True,
) -> Tuple[int, int, int]:
    """Load dataset, optionally restrict classes, create splits, and save to disk."""
    print(f"Loading dataset '{dataset_name}' split='{split}'...")
    base = load_dataset(dataset_name, split=split)
    label_col = detect_label_column(base)
    print(f"Detected label column: {label_col}")
    print(f"Total samples before filtering: {len(base)}")

    if top_n_classes is not None:
        print(f"Restricting to top {top_n_classes} classes...")
        base = restrict_to_top_n_classes(base, label_col, top_n_classes)
        print(f"Samples after top-{top_n_classes} filtering: {len(base)}")

    if shuffle_before:
        print("Shuffling dataset before splitting...")
        base = base.shuffle(seed=seed)

    if stratify:
        print(f"Creating stratified splits {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%} ...")
        dd = stratified_splits(base, label_col, train_ratio, val_ratio, test_ratio, seed)
    else:
        print(f"Creating random splits {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%} ...")
        dd = random_splits(base, train_ratio, val_ratio, test_ratio, seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving splits to: {out_dir}")
    dd.save_to_disk(str(out_dir))

    n_train, n_val, n_test = len(dd['train']), len(dd['validation']), len(dd['test'])
    print(f"Saved. Sizes -> train: {n_train}, val: {n_val}, test: {n_test}")
    return n_train, n_val, n_test


def main():
    parser = argparse.ArgumentParser(description="Prepare stratified train/val/test splits and save to disk.")
    parser.add_argument("--dataset", type=str, default="pvrancx/legobricks", help="HF dataset name")
    parser.add_argument("--split", type=str, default="train", help="Source split to partition")
    parser.add_argument("--output", type=str, default="data/splits/legobricks", help="Output directory to save splits")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio (0-1)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (0-1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio (0-1)")
    parser.add_argument("--top_n_classes", type=int, default=None, help="Restrict to top-N most frequent classes")
    parser.add_argument("--no_stratify", action="store_true", help="Disable stratification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffle before splitting")
    args = parser.parse_args()

    prepare_splits(
        dataset_name=args.dataset,
        split=args.split,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        top_n_classes=args.top_n_classes,
        stratify=not args.no_stratify,
        seed=args.seed,
        shuffle_before=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()


