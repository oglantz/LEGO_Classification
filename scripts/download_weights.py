#!/usr/bin/env python3
"""Download FastSAM model weights."""

import argparse
import sys
from pathlib import Path
import urllib.request
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_file(url: str, output_path: Path, description: str = "file"):
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        description: Description of file for progress messages
    """
    print(f"Downloading {description} from {url}...")
    print(f"Saving to {output_path}...")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print(f"\n✓ Successfully downloaded {description}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {description}: {e}")
        return False


def download_fastsam_weights(model_type: str = "FastSAM-x"):
    """
    Download FastSAM model weights.
    
    Args:
        model_type: Model type ("FastSAM-x" or "FastSAM-s")
    """
    # Create models directory
    models_dir = Path("models/segmentation")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # FastSAM model URLs (these are example URLs - actual URLs may vary)
    # Users may need to download from official FastSAM repository
    model_urls = {
        "FastSAM-x": "https://github.com/CASIA-IVA-Lab/FastSAM/raw/main/weights/FastSAM-x.pt",
        "FastSAM-s": "https://github.com/CASIA-IVA-Lab/FastSAM/raw/main/weights/FastSAM-s.pt",
    }
    
    if model_type not in model_urls:
        print(f"Error: Unknown model type {model_type}")
        print(f"Available types: {list(model_urls.keys())}")
        return False
    
    output_path = models_dir / f"{model_type}.pt"
    
    # Check if already exists
    if output_path.exists():
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return True
    
    # Download
    url = model_urls[model_type]
    success = download_file(url, output_path, f"{model_type} weights")
    
    if success:
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        print(f"\n✓ FastSAM weights downloaded successfully!")
        print(f"  Location: {output_path}")
        return True
    else:
        print("\n✗ Failed to download FastSAM weights.")
        print("\nAlternative download methods:")
        print("1. Install FastSAM package: pip install fastsam")
        print("2. Download manually from: https://github.com/CASIA-IVA-Lab/FastSAM")
        print(f"3. Place weights in: {models_dir}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download FastSAM model weights"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="FastSAM-x",
        choices=["FastSAM-x", "FastSAM-s"],
        help="Model type to download (default: FastSAM-x)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FastSAM Weight Downloader")
    print("=" * 60)
    print(f"\nModel type: {args.model_type}")
    print("\nNote: This script attempts to download weights from the official")
    print("FastSAM repository. If download fails, you can:")
    print("  1. Install FastSAM: pip install fastsam")
    print("  2. Download manually from GitHub")
    print("  3. Use the FastSAM package's built-in download")
    print()
    
    success = download_fastsam_weights(args.model_type)
    
    if success:
        print("\n" + "=" * 60)
        print("Setup complete!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Download failed. Please use alternative methods.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

