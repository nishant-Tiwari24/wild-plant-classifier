#!/usr/bin/env python3
"""
Test the pre-trained models with sample images
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np

# Plant class names
LABELS = ['alfalfa', 'allium', 'borage', 'burdock', 'calendula', 'cattail', 
          'chickweed', 'chicory', 'chive_blossom', 'coltsfoot', 'common_mallow', 
          'common_milkweed', 'common_vetch', 'common_yarrow', 'coneflower', 
          'cow_parsely', 'cowslip', 'crimson_clover', 'crithmum_maritimum', 
          'daisy', 'dandelion', 'fennel', 'firewood', 'gardenia', 'garlic_mustard', 
          'geranium', 'ground_ivy', 'harebell', 'henbit', 'knapweed', 
          'meadowsweet', 'mullein', 'pickerelweed', 'ramsons', 'red_clover']

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, model_type='resnet34'):
    """Load a pre-trained model"""
    print(f"\nLoading {model_type} model...")
    
    if model_type == 'resnet34':
        model = models.resnet34(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 35)
        )
    elif model_type == 'googlenet':
        model = models.googlenet(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 35)
        )
    elif model_type == 'mobilenet':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 35)
        )
    
    # Load weights (with weights_only=False for older PyTorch models)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model

def predict_image(model, image_path, top_k=5):
    """Make predictions on an image"""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    return top_probs.numpy(), top_indices.numpy()

def main():
    print("=" * 70)
    print("Wild Edible Plant Classifier - Model Testing")
    print("=" * 70)
    
    # Check for sample images
    sample_dir = 'dataset/sample'
    if not os.path.exists(sample_dir):
        print("\nError: Sample directory not found!")
        return
    
    # Get first available image from each class
    sample_images = []
    for class_name in LABELS[:5]:  # Test first 5 classes
        class_dir = os.path.join(sample_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                sample_images.append((class_name, os.path.join(class_dir, images[0])))
    
    if not sample_images:
        print("\nNo sample images found!")
        return
    
    print(f"\nFound {len(sample_images)} sample images to test")
    
    # Load ResNet-34 model
    model_path = 'saved_models/best_resnet34.pt'
    if not os.path.exists(model_path):
        print(f"\nModel not found: {model_path}")
        return
    
    model = load_model(model_path, 'resnet34')
    
    # Test predictions
    print("\n" + "=" * 70)
    print("Making Predictions...")
    print("=" * 70)
    
    for true_class, image_path in sample_images:
        print(f"\n{'─' * 70}")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"True Class: {true_class}")
        print(f"{'─' * 70}")
        
        probs, indices = predict_image(model, image_path, top_k=5)
        
        print("\nTop 5 Predictions:")
        for i, (prob, idx) in enumerate(zip(probs, indices), 1):
            predicted_class = LABELS[idx]
            marker = "✓" if predicted_class == true_class else " "
            print(f"  {marker} {i}. {predicted_class:20s} - {prob*100:5.2f}%")
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print("\nNote: For full training and evaluation, open the Jupyter notebooks")
    print("in JupyterLab at: http://localhost:8888")
    print("=" * 70)

if __name__ == "__main__":
    main()
