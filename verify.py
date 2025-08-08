#!/usr/bin/env python3
"""
Verification script for twin face verification
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from models.modeling import VisionTransformer, CONFIGS
import numpy as np
import argparse
import os

def load_model(checkpoint_path, model_type="ViT-B_16", img_size=448):
    """Load trained model"""
    config = CONFIGS[model_type]
    model = VisionTransformer(config, img_size=img_size, num_classes=2)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def preprocess_image(image_path, img_size=448):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def verify_pair(model, img1_path, img2_path, threshold=0.0):
    """Verify if two images are of the same person"""
    # Preprocess images
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)
    
    # Get embeddings
    with torch.no_grad():
        emb1 = model(img1, return_embedding=True)
        emb2 = model(img2, return_embedding=True)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=1).item()
        
        # Make prediction
        is_same = similarity > threshold
        
    return similarity, is_same

def main():
    parser = argparse.ArgumentParser(description='Twin Face Verification')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--img1', required=True, help='Path to first image')
    parser.add_argument('--img2', required=True, help='Path to second image')
    parser.add_argument('--threshold', type=float, default=0.0, help='Similarity threshold')
    parser.add_argument('--model_type', default='ViT-B_16', help='Model type')
    parser.add_argument('--img_size', type=int, default=448, help='Image size')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.model_type, args.img_size)
    
    # Verify pair
    similarity, is_same = verify_pair(model, args.img1, args.img2, args.threshold)
    
    print(f"Similarity: {similarity:.4f}")
    print(f"Same person: {is_same}")
    print(f"Threshold: {args.threshold}")

if __name__ == "__main__":
    main() 