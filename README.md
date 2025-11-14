# CALM: Class-anchor-ALigned generative Modeling

PyTorch implementation of the paper **"Generative Modeling of Class Probability for Multi-Modal Representation Learning"**.

## Overview

CALM is a novel approach for multi-modal representation learning that addresses modality discrepancies and information imbalance between video and text. The key innovations include:

1. **Class Anchor Alignment**: Uses class labels from an independent dataset as semantic anchors
2. **Probability Distribution Modeling**: Computes probability distributions between modalities and class anchors
3. **Cross-Modal Probabilistic VAE**: Models uncertainty and captures deeper relationships between modalities

## Architecture

```
Video Frames → CLIP Encoder → Video Features ──┐
                                                 ├→ Probability Distributions → VAE → Alignment
Text → CLIP Encoder → Text Features ────────────┘
                                                 ↑
Class Anchors ──────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd major_fmg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install CLIP:
```bash
pip install git+https://github.com/openai/CLIP.git
```

## Project Structure

```
major_fmg/
├── calm/
│   ├── models/
│   │   ├── calm.py                    # Main CALM model
│   │   ├── class_anchors.py           # Class anchor extraction
│   │   ├── probability_distribution.py # Probability distribution computation
│   │   ├── cross_modal_vae.py         # Cross-modal VAE
│   │   └── losses.py                  # Loss functions
│   ├── data/
│   │   └── video_text_dataset.py      # Dataset loaders
│   ├── utils/
│   │   ├── helpers.py                 # Helper utilities
│   │   └── metrics.py                 # Evaluation metrics
│   └── configs/
│       ├── base_config.py             # Base configuration
│       └── msrvtt_config.py           # Dataset-specific configs
├── scripts/
│   ├── train.py                       # Training script
│   └── evaluate.py                    # Evaluation script
├── datasets/                          # Place your datasets here
├── checkpoints/                       # Model checkpoints
└── outputs/                           # Training outputs
```

## Quick Start

### 1. Prepare Dataset

Download one of the supported datasets:
- **MSR-VTT**: [Download link](http://ms-multimedia-challenge.com/2017/dataset)
- **DiDeMo**: [Download link](https://github.com/LisaAnne/LocalizingMoments)
- **MSVD**: [Download link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
- **LSMDC**: [Download link](https://sites.google.com/site/describingmovies/)

Organize your dataset as:
```
datasets/
└── msrvtt/
    ├── videos/
    │   ├── video0.mp4
    │   ├── video1.mp4
    │   └── ...
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json
```

### 2. Training

Train CALM on MSR-VTT:

```bash
python scripts/train.py \
    --dataset msrvtt \
    --data_root ./datasets/msrvtt/videos \
    --annotation_train ./datasets/msrvtt/annotations/train.json \
    --annotation_val ./datasets/msrvtt/annotations/val.json \
    --batch_size 128 \
    --num_epochs 5 \
    --lr 1e-5 \
    --alpha 0.1 \
    --output_dir ./outputs \
    --experiment_name calm_msrvtt \
    --num_workers 4
```

**Key Arguments:**
- `--dataset`: Dataset name (msrvtt, didemo, msvd, lsmdc)
- `--data_root`: Directory containing video files
- `--annotation_train/val`: Paths to annotation files
- `--batch_size`: Batch size (default: 128)
- `--num_epochs`: Number of training epochs (5 for retrieval, 20 for captioning)
- `--lr`: Learning rate (default: 1e-5)
- `--alpha`: Weight for KL divergence loss (default: 0.1)
- `--num_frames`: Number of frames to sample (default: 12)
- `--latent_dim`: VAE latent dimension (default: 256)
- `--temperature`: Temperature for probability distributions (default: 0.07)

### 3. Evaluation

Evaluate trained model:

```bash
python scripts/evaluate.py \
    --dataset msrvtt \
    --data_root ./datasets/msrvtt/videos \
    --annotation_file ./datasets/msrvtt/annotations/test.json \
    --checkpoint ./outputs/calm_msrvtt/checkpoints/best_model.pth \
    --batch_size 32 \
    --output_file ./outputs/calm_msrvtt/results/test_results.json
```

### 4. Using the Model Programmatically

```python
import torch
import clip
from calm.models import CALMForRetrieval
from calm.models.class_anchors import load_class_labels

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load class labels
class_labels = load_class_labels('charades')

# Create CALM model
model = CALMForRetrieval(
    clip_model=clip_model,
    tokenizer=clip.tokenize,
    class_labels=class_labels,
    num_frames=12,
    latent_dim=256,
    temperature=0.07,
    device=device
)

# Load trained weights
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
video_frames = ...  # [batch_size, num_frames, 3, 224, 224]
text_tokens = clip.tokenize(["a person is walking"])

with torch.no_grad():
    video_embeddings = model.get_video_embeddings(video_frames)
    text_embeddings = model.get_text_embeddings(text_tokens)

    # Compute similarity
    similarity = torch.matmul(video_embeddings, text_embeddings.t())
```

## Model Components

### 1. Class Anchor Extraction

```python
from calm.models.class_anchors import ClassAnchorExtractor, load_class_labels

# Load Charades action classes (157 classes)
class_labels = load_class_labels('charades')

# Create anchor extractor
anchor_extractor = ClassAnchorExtractor(
    class_labels=class_labels,
    text_encoder=clip_model.encode_text,
    prompt_template="The content of {}"
)

# Get class anchors
class_anchors = anchor_extractor()  # [157, 512]
```

### 2. Probability Distribution Computation

```python
from calm.models.probability_distribution import ProbabilityDistributionComputer

prob_computer = ProbabilityDistributionComputer(temperature=0.07)

# Compute probability distributions
video_prob, text_prob = prob_computer(
    video_features,    # [batch_size, 512]
    text_features,     # [batch_size, 512]
    class_anchors      # [157, 512]
)
# Returns: [batch_size, 157] probability distributions
```

### 3. Cross-Modal VAE

```python
from calm.models.cross_modal_vae import CrossModalVAE

vae = CrossModalVAE(
    num_classes=157,
    latent_dim=256,
    hidden_dims=[512, 384],
    dropout=0.1
)

# Reconstruct text distribution from video distribution
outputs = vae(video_prob_dist)
# Returns: {
#   'recon_text_prob': reconstructed distribution,
#   'mu': latent mean,
#   'logvar': latent log variance,
#   'z': latent variable
# }
```

## Configuration

Modify hyperparameters in `calm/configs/base_config.py` or create custom configs:

```python
from calm.configs import BaseConfig

class CustomConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Override parameters
        self.TRAIN.update({
            'batch_size': 64,
            'num_epochs': 10,
            'learning_rate': 5e-6
        })

        self.MODEL.update({
            'latent_dim': 512,
            'temperature': 0.05
        })
```

## Supported Datasets

| Dataset | Videos | Captions | Domain | Annotation Format |
|---------|--------|----------|--------|-------------------|
| MSR-VTT | 10,000 | 200,000 | General | JSON |
| DiDeMo | 10,642 | 40,543 | Personal | JSON |
| MSVD | 1,970 | ~80,000 | General | JSON |
| LSMDC | 118,081 | 118,081 | Movies | TSV |

## Expected Results

Based on the paper, here are the expected performance metrics:

### MSR-VTT (In-domain)

| Metric | R@1 | R@5 | R@10 | MnR |
|--------|-----|-----|------|-----|
| Text-to-Video | 50.8 | 77.5 | 85.8 | 11.7 |

### DiDeMo (Out-of-domain, trained on MSR-VTT)

| Metric | R@1 | R@5 | R@10 | MnR |
|--------|-----|-----|------|-----|
| Text-to-Video | 41.2 | 66.3 | 76.3 | 16.1 |

## Key Hyperparameters (from paper)

- **CLIP Model**: ViT-B/32
- **Number of Frames**: 12
- **Batch Size**: 128
- **Learning Rate**: 1e-5
- **Optimizer**: AdamW
- **Epochs**: 5 (retrieval), 20 (captioning)
- **VAE Latent Dim**: 256
- **Temperature (τ)**: 0.07
- **KL Weight (α)**: 0.1
- **Dropout**: 0.1

## Citation

```bibtex
@article{shin2025calm,
  title={Generative Modeling of Class Probability for Multi-Modal Representation Learning},
  author={Shin, JungKyoo and Kim, Bumsoo and Kim, Eunwoo},
  journal={arXiv preprint arXiv:2503.17417},
  year={2025}
}
```

## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- Implementation based on the paper by Shin et al.

## License

This implementation is for research purposes only.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or number of frames
2. **CLIP Installation**: Make sure to install from git repository
3. **Video Loading**: Ensure opencv-python is properly installed
4. **Slow Data Loading**: Increase `num_workers` parameter

### Tips

- Use mixed precision training for faster training: `torch.cuda.amp`
- Pre-extract features for faster iteration during development
- Monitor GPU memory usage with `nvidia-smi`

## Contact

For questions or issues, please open an issue on GitHub.
