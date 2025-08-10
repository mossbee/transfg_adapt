# TransFG for Identical Twin Face Verification

This repository adapts the TransFG (Transformer for Fine-Grained Recognition) architecture for the task of identical twin face verification. The model uses a Vision Transformer with Part Selection Module and contrastive learning to distinguish between identical twins.

## Overview

The adaptation includes:
- **Vision Transformer backbone** with overlapping patch embedding
- **Part Selection Module (PSM)** to focus on discriminative facial regions
- **Contrastive learning** to maximize differences between different twins while minimizing differences for the same person
- **Verification-specific loss function** optimized for face verification tasks

## Dataset

The model is trained on the NDTWIN dataset, which contains pairs of face images with labels indicating whether they belong to the same person (1) or different people (0).

### Dataset Format
- Training pairs: `train_pairs_dataset.txt`
- Testing pairs: `test_pairs_dataset.txt`
- Each line: `image1.jpg image2.jpg label`

### Dataset Structure
```
Dataset/
├── ND_TWIN_448/
│   ├── 90004/
│   │   ├── 90004d43.jpg
│   │   ├── 90004d65.jpg
│   │   └── ...
│   └── ...
├── train_pairs_dataset.txt
└── test_pairs_dataset.txt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TransFG
```

2. Install dependencies:
```bash
pip install torch torchvision scikit-learn tqdm tensorboard pillow numpy
```

3. Download pretrained ViT weights:
```bash
mkdir pretrained
# Download ViT-B_16.npz to pretrained/ directory
```

## Training

### Basic Training Command
```bash
python train.py \
    --name "twin_verification_run" \
    --dataset NDTWIN \
    --data_root /path/to/dataset \
    --model_type ViT-B_16 \
    --pretrained_dir pretrained/ViT-B_16.npz \
    --output_dir ./output/NDTWIN \
    --img_size 448 \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --learning_rate 3e-2 \
    --num_steps 1000 \
    --margin 0.4 \
    --similarity_threshold 0.0
```

### Key Parameters

- `--margin`: Margin for contrastive loss (default: 0.4)
- `--similarity_threshold`: Threshold for similarity-based prediction during training (default: 0.0)
- `--img_size`: Input image size (default: 448)
- `--model_type`: ViT model variant (ViT-B_16, ViT-B_32, ViT-L_16, etc.)

### Training Features

1. **Contrastive Loss**: Minimizes distance between embeddings of the same person while maximizing distance between different people
2. **Data Augmentation**: Random crop, horizontal flip, color jitter, and rotation for robust training
3. **Mixed Precision Training**: Optional FP16 training for faster training
4. **Gradient Accumulation**: Supports training with larger effective batch sizes

## Evaluation

### Evaluate on Test Set
```bash
python verify.py \
    --model_path ./output/NDTWIN/twin_verification_run_checkpoint.bin \
    --model_type ViT-B_16 \
    --data_root /path/to/dataset \
    --evaluate
```

### Verify Specific Image Pair
```bash
python verify.py \
    --model_path ./output/NDTWIN/twin_verification_run_checkpoint.bin \
    --model_type ViT-B_16 \
    --img1_path /path/to/image1.jpg \
    --img2_path /path/to/image2.jpg \
    --threshold 0.0
```

## Model Architecture

### Vision Transformer Backbone
- Uses overlapping patch embedding to preserve local information
- Multi-head self-attention mechanism
- Position embeddings for spatial information

### Part Selection Module (PSM)
- Integrates attention weights from all layers
- Selects most discriminative tokens for final classification
- Focuses on subtle differences between identical twins

### Verification Head
- Extracts embeddings from CLS token
- Uses cosine similarity for verification
- Contrastive learning for better feature separation

## Key Improvements

1. **Proper Verification Loss**: Replaced BCE loss with contrastive loss suitable for verification tasks
2. **Enhanced Data Preprocessing**: Added comprehensive augmentation for face verification
3. **Similarity-based Training**: Direct similarity optimization instead of classification
4. **Proper Metrics**: EER, ROC AUC, and optimal threshold calculation

## Results

The model achieves:
- **EER (Equal Error Rate)**: Measures verification accuracy
- **ROC AUC**: Area under ROC curve for verification performance
- **Optimal Threshold**: Automatically determined threshold for best performance

## Usage Examples

### Training with Custom Parameters
```bash
python train.py \
    --name "custom_run" \
    --dataset NDTWIN \
    --data_root /path/to/dataset \
    --model_type ViT-L_16 \
    --img_size 512 \
    --train_batch_size 16 \
    --learning_rate 1e-2 \
    --margin 0.5 \
    --num_steps 2000
```

### Batch Verification
```python
from verify import load_model, verify_pair
from models.modeling import CONFIGS

# Load model
config = CONFIGS['ViT-B_16']
model = load_model('checkpoint.bin', config)

# Verify multiple pairs
pairs = [
    ('image1.jpg', 'image2.jpg'),
    ('image3.jpg', 'image4.jpg'),
    # ...
]

for img1_path, img2_path in pairs:
    similarity, is_same = verify_pair(model, img1_path, img2_path)
    print(f"{img1_path} vs {img2_path}: {similarity:.4f} -> Same: {is_same}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Poor Performance**: Adjust margin and learning rate
3. **Dataset Loading Errors**: Check data paths and file permissions

### Performance Tips

1. Use larger models (ViT-L_16, ViT-H_14) for better accuracy
2. Increase training steps for convergence
3. Tune margin parameter based on dataset characteristics
4. Use data augmentation for better generalization

## Citation

If you use this code, please cite the original TransFG paper:

```bibtex
@inproceedings{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-Grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

