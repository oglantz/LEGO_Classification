# LEGO Piece Detection and Classification System

A computer vision system that takes an overhead photo of LEGO pieces spread on a table and returns:
- A list of detected pieces (part IDs / class IDs)
- Counts per piece
- An annotated output image showing each detected instance with its predicted ID and confidence

## Architecture

This system implements a **two-stage pipeline**:

### Stage A: Instance Segmentation
- Uses **FastSAM** (lightweight Segment Anything Model) for automatic mask generation
- Post-processes masks to remove noise, merge overlaps, and filter by size/aspect ratio

### Stage B: Per-Instance Classification
- Uses **ResNet50** fine-tuned on LEGO parts dataset for classification
- Applies geometric re-ranking using aspect ratio, area, and stud count heuristics
- Supports top-k predictions with confidence scores

### Pipeline Flow

```
Input Image
    ↓
FastSAM Segmentation → Mask Post-processing
    ↓
Extract Instances
    ↓
ResNet50 Classification → Geometric Re-ranking
    ↓
JSON Output + Annotated Image
```

## Repository Structure

```
ozl/
├── src/
│   ├── segmentation/          # Instance segmentation (FastSAM)
│   ├── classification/         # Per-instance classification (ResNet50)
│   ├── postprocessing/         # Geometric re-ranking
│   ├── utils/                  # Common utilities
│   └── visualization/          # Annotation and visualization
├── scripts/
│   ├── run_inference.py        # Main inference script
│   ├── train_classifier.py     # Training script
│   └── download_weights.py     # Download FastSAM weights
├── configs/
│   ├── default.yaml            # Main configuration
│   └── augmentation.yaml       # Augmentation parameters
├── models/                     # Model weights (gitignored)
│   ├── segmentation/           # FastSAM weights
│   └── classification/      # Trained ResNet50 checkpoints
├── outputs/                    # Inference outputs (gitignored)
├── data/                       # Dataset storage
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training, CPU fallback available)
- NVIDIA drivers installed (check with `nvidia-smi`)
- ~10GB disk space for models and datasets

### Installation

1. **Clone the repository** (if applicable):
```bash
cd ozl
```

2. **Install PyTorch with CUDA support** (recommended for GPU acceleration):

First, check your CUDA version:
```bash
nvidia-smi
```

Then install PyTorch with the appropriate CUDA version:

**For CUDA 12.x (most modern GPUs):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (not recommended for training):**
```bash
pip install torch torchvision torchaudio
```

Verify CUDA is available:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

3. **Install other dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install FastSAM** (if not already installed):
```bash
pip install fastsam
```

Or install from source:
```bash
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
cd FastSAM
pip install -e .
```

4. **Download FastSAM weights** (optional, will be downloaded automatically if needed):
```bash
python scripts/download_weights.py --model-type FastSAM-x
```

> **Note**: Training on CPU is extremely slow (~10 seconds per batch). GPU acceleration is strongly recommended for training.

## Quick Start

### Minimal Viable Demo

Even without a trained classifier, the pipeline runs end-to-end using:
- FastSAM for segmentation (pretrained, works immediately)
- Dummy classifier (ImageNet-pretrained ResNet50 with random head)

**Run inference on an image**:
```bash
python scripts/run_inference.py --image path/to/your/image.jpg --output outputs/
```

This will:
1. Segment the image using FastSAM
2. Classify each instance (using dummy classifier if no trained model)
3. Generate JSON results and annotated image in `outputs/`

### Training the Classifier

1. **Train the ResNet50 classifier**:
```bash
python scripts/train_classifier.py --config configs/default.yaml
```

This will:
- Download the HuggingFace dataset (`pvrancx/legobricks`) automatically
- Train ResNet50 on LEGO parts
- Save checkpoints to `models/classification/`
- Log metrics to TensorBoard in `logs/`

2. **Monitor training**:
```bash
tensorboard --logdir logs
```

3. **Run inference with trained model**:
```bash
python scripts/run_inference.py \
    --image path/to/your/image.jpg \
    --checkpoint models/classification/best_checkpoint.pt \
    --output outputs/
```

## Usage

### Inference

```bash
python scripts/run_inference.py \
    --image <input_image_path> \
    [--config <config_path>] \
    [--output <output_dir>] \
    [--checkpoint <checkpoint_path>]
```

**Arguments**:
- `--image`: Path to input image (required)
- `--config`: Path to config file (default: `configs/default.yaml`)
- `--output`: Output directory (default: `outputs/`)
- `--checkpoint`: Path to classifier checkpoint (overrides config)

**Outputs**:
- `<image_name>_results.json`: JSON with all predictions and summary
- `<image_name>_annotated.png`: Annotated image with masks and labels

### Training

```bash
python scripts/train_classifier.py \
    [--config <config_path>] \
    [--resume <checkpoint_path>]
```

**Arguments**:
- `--config`: Path to config file (default: `configs/default.yaml`)
- `--resume`: Path to checkpoint to resume from

### Configuration

Edit `configs/default.yaml` to customize:
- Segmentation parameters (mask thresholds, overlap handling)
- Classification settings (number of classes, confidence thresholds)
- Training hyperparameters (learning rate, epochs, batch size)
- Visualization options

## Output Format

### JSON Output

```json
{
  "image_path": "path/to/image.jpg",
  "num_instances": 15,
  "instances": [
    {
      "instance_id": 0,
      "bbox": [100, 200, 150, 250],
      "predicted_id": 1234,
      "confidence": 0.85,
      "top_k_predictions": [
        [1234, 0.85],
        [1235, 0.12],
        ...
      ],
      "geometric_features": {
        "aspect_ratio": 1.2,
        "area": 2500,
        "stud_count_proxy": 0.3
      }
    },
    ...
  ],
  "summary": {
    "1234": 5,
    "5678": 3,
    ...
  }
}
```

### Annotated Image

The annotated image shows:
- Colored masks for each detected instance
- Bounding boxes
- Labels with part ID and confidence score

## Dataset

The system uses the **pvrancx/legobricks** dataset from HuggingFace, which contains:
- Rendered images of individual LEGO parts
- Part IDs as labels
- Multiple views per part

The dataset is downloaded automatically during training.

## Limitations & Mitigation

### Current Limitations

**Without custom labeled table-scene images**:
- **Domain gap**: Synthetic renderings → real photos
  - *Mitigation*: Heavy augmentations (blur, noise, color shifts, backgrounds)
- **Occlusion handling**: Limited ability to detect partially visible pieces
  - *Mitigation*: Multi-scale detection, attention mechanisms (future)
- **Background confusion**: May confuse background with pieces
  - *Mitigation*: Better segmentation post-processing, background removal
- **Lighting variations**: May affect accuracy
  - *Mitigation*: Augmentations, normalization, domain adaptation

### Pseudo-labeling Strategy

To reduce manual labeling:
1. Run inference on unlabeled photos
2. Manually verify/correct predictions
3. Use verified data to retrain model
4. Iterate to improve accuracy

## Roadmap

1. **MVP** ✅
   - Working pipeline with FastSAM + ResNet50
   - End-to-end inference
   - Basic training loop

2. **Improved Segmentation** (Next)
   - Fine-tune FastSAM on LEGO-specific data
   - Better mask post-processing
   - Handle overlapping pieces

3. **Domain Adaptation**
   - Adversarial training
   - Style transfer
   - Real-world data augmentation

4. **Expand Classes**
   - Scale to 5000+ parts
   - Hierarchical classification
   - Few-shot learning for rare parts

5. **Handle Occlusion**
   - Multi-scale detection
   - Attention mechanisms
   - Partial visibility handling

6. **Multi-view/Video**
   - Temporal consistency
   - 3D reconstruction
   - Video processing

## Troubleshooting

### FastSAM not found
```bash
pip install fastsam
# Or download weights manually
python scripts/download_weights.py
```

### CUDA not available / Training on CPU
If you see "CUDA not available, falling back to CPU", PyTorch was installed without CUDA support.

1. Check your GPU and CUDA version:
```bash
nvidia-smi
```

2. Check current PyTorch installation:
```bash
python -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"
```

3. If it shows `+cpu` in the version, reinstall with CUDA:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA out of memory
- Reduce batch size in config
- Use CPU mode: set `device.use_cuda: false` in config
- Use smaller model: `FastSAM-x` instead of `FastSAM-s`

### Dataset download fails
- Check internet connection
- Verify HuggingFace dataset name: `pvrancx/legobricks`
- Try downloading manually from HuggingFace

### No masks detected
- Check image quality and lighting
- Adjust segmentation thresholds in config
- Try different `points_per_side` values
- Verify FastSAM weights are downloaded

## Image Capture Protocol

For best results when capturing photos:

1. **Background**: Use a solid, contrasting color (white or light gray recommended)
2. **Lighting**: Even, diffused lighting to minimize shadows
3. **Camera angle**: Overhead, perpendicular to table
4. **Distance**: Fill frame with pieces, maintain focus
5. **Resolution**: At least 1920x1080, higher is better
6. **Format**: JPEG or PNG

## Contributing

This is a starter repository. To extend:

1. Add new segmentation models (YOLO, Mask R-CNN)
2. Implement better geometric heuristics
3. Add support for video input
4. Improve domain adaptation techniques
5. Expand to more LEGO part classes

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- FastSAM: https://github.com/CASIA-IVA-Lab/FastSAM
- HuggingFace Datasets: https://huggingface.co/datasets
- PyTorch and torchvision

## Contact

For questions or issues, please open an issue in the repository.
