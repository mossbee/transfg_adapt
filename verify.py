#!/usr/bin/env python3
"""
Verification script for TransFG twin verification model
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import get_loader
from train import calculate_verification_metrics, verification_loss

def load_model(model_path, config, img_size=448, device='cuda'):
    """Load trained model"""
    model = VisionTransformer(config, img_size, zero_head=True, num_classes=2)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path, img_size=448):
    """Preprocess single image for verification"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def verify_pair(model, img1_path, img2_path, device='cuda', threshold=0.0):
    """Verify if two images are of the same person"""
    model.eval()
    
    # Preprocess images
    img1 = preprocess_image(img1_path).to(device)
    img2 = preprocess_image(img2_path).to(device)
    
    with torch.no_grad():
        # Get embeddings
        emb1 = model(img1, return_embedding=True)
        emb2 = model(img2, return_embedding=True)
        
        # Calculate similarity
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        similarity = torch.sum(emb1 * emb2, dim=1).item()
        
        # Make prediction
        is_same = similarity > threshold
        
    return similarity, is_same

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    all_similarities, all_labels = [], []
    
    print("Evaluating model...")
    for batch in test_loader:
        img1, img2, labels = [t.to(device) for t in batch]
        
        with torch.no_grad():
            emb1 = model(img1, return_embedding=True)
            emb2 = model(img2, return_embedding=True)
            
            # Calculate similarity
            emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
            emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
            similarities = torch.sum(emb1 * emb2, dim=1)
            
            all_similarities.append(similarities.cpu())
            all_labels.append(labels.cpu())
    
    all_similarities = torch.cat(all_similarities, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    metrics = calculate_verification_metrics(all_similarities, all_labels)
    
    print(f"Evaluation Results:")
    print(f"EER: {metrics['eer']:.4f}")
    print(f"ROC AUC: {metrics['auc']:.4f}")
    print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='TransFG Twin Verification')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_type', default='ViT-B_16', 
                        choices=['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14'],
                        help='Model type')
    parser.add_argument('--img_size', type=int, default=448,
                        help='Image size')
    parser.add_argument('--data_root', type=str, default='/home/mossbee/Work/Dataset',
                        help='Data root directory')
    parser.add_argument('--img1_path', type=str, default=None,
                        help='Path to first image for verification')
    parser.add_argument('--img2_path', type=str, default=None,
                        help='Path to second image for verification')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on test set')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Similarity threshold for verification')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    config = CONFIGS[args.model_type]
    model = load_model(args.model_path, config, args.img_size, device)
    print(f"Model loaded from {args.model_path}")
    
    if args.evaluate:
        # Evaluate on test set
        from utils.data_utils import get_loader
        import sys
        sys.argv = ['train.py', '--dataset', 'NDTWIN', '--data_root', args.data_root]
        
        # Create a mock args object for get_loader
        class MockArgs:
            def __init__(self):
                self.dataset = 'NDTWIN'
                self.data_root = args.data_root
                self.img_size = args.img_size
                self.train_batch_size = 4
                self.eval_batch_size = 4
        
        mock_args = MockArgs()
        _, test_loader = get_loader(mock_args)
        
        metrics = evaluate_model(model, test_loader, device)
        
    elif args.img1_path and args.img2_path:
        # Verify specific image pair
        similarity, is_same = verify_pair(model, args.img1_path, args.img2_path, device, args.threshold)
        
        print(f"Similarity: {similarity:.4f}")
        print(f"Same person: {is_same}")
        print(f"Threshold used: {args.threshold}")
        
    else:
        print("Please specify either --evaluate or both --img1_path and --img2_path")

if __name__ == '__main__':
    main() 