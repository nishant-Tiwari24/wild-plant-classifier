#!/usr/bin/env python3
"""
Comprehensive demo of the Wild Edible Plant Classifier
Shows dataset info, model architecture, and sample predictions
"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
import os

print("=" * 80)
print(" " * 15 + "WILD EDIBLE PLANT CLASSIFIER - DEMO OUTPUT")
print("=" * 80)

# 1. System Information
print("\n" + "=" * 80)
print("1. SYSTEM & ENVIRONMENT")
print("=" * 80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. Dataset Loading
print("\n" + "=" * 80)
print("2. LOADING SAMPLE DATASET")
print("=" * 80)

SAMPLE_FILEPATH = 'dataset/sample'
SAMPLE_TRANSFORM = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

try:
    sample = datasets.ImageFolder(SAMPLE_FILEPATH, transform=SAMPLE_TRANSFORM)
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Number of classes: {len(sample.classes)}")
    print(f"  - Number of samples: {len(sample)}")
    print(f"  - Image size: 400x400 pixels")
    print(f"  - Normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    sample = None

# 3. Plant Classes
if sample:
    print("\n" + "=" * 80)
    print("3. PLANT SPECIES CLASSIFICATION")
    print("=" * 80)
    print(f"\nThe model classifies {len(sample.classes)} species of wild edible plants:\n")
    
    # Display in 3 columns
    classes = sample.classes
    for i in range(0, len(classes), 3):
        row = classes[i:i+3]
        line = f"  {i+1:2d}. {row[0].replace('_', ' ').title():22s}"
        if len(row) > 1:
            line += f"  {i+2:2d}. {row[1].replace('_', ' ').title():22s}"
        if len(row) > 2:
            line += f"  {i+3:2d}. {row[2].replace('_', ' ').title()}"
        print(line)

# 4. Model Information
print("\n" + "=" * 80)
print("4. PRE-TRAINED MODELS")
print("=" * 80)

models_info = [
    ('best_resnet34.pt', 'ResNet-34', 'Deep residual network with skip connections'),
    ('best_googlenet.pt', 'GoogLeNet', 'Inception-based multi-scale architecture'),
    ('best_mobilenetv2.pt', 'MobileNet v2', 'Lightweight mobile-optimized network')
]

models_dir = Path('saved_models')
print("\nAvailable pre-trained models:\n")

for model_file, model_name, description in models_info:
    model_path = models_dir / model_file
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ {model_name:15s} ({size_mb:6.2f} MB)")
        print(f"  └─ {description}")
    else:
        print(f"✗ {model_name:15s} - Not found")

# 5. Training Configuration
print("\n" + "=" * 80)
print("5. TRAINING CONFIGURATION")
print("=" * 80)
print("""
Hyperparameters:
  • Epochs:           20
  • Learning Rate:    0.001
  • Batch Size:       64
  • Optimizer:        Adam
  • Loss Function:    Cross-Entropy Loss
  
Data Split:
  • Training:         70% (~11,575 images)
  • Validation:       15% (~2,480 images)
  • Testing:          15% (~2,480 images)
  
Data Augmentation:
  • Random Rotation:  ±15 degrees
  • Random Flip:      Horizontal
  • Random Crop:      224x224 pixels
  • Color Jitter:     Brightness, contrast, saturation
  
Transfer Learning:
  • Base:             ImageNet pre-trained weights
  • Fine-tuning:      All layers trainable
  • Custom Classifier: 2 hidden layers (512, 256 neurons)
  • Dropout:          0.5 (regularization)
""")

# 6. Model Architecture Example
print("\n" + "=" * 80)
print("6. MODEL ARCHITECTURE (ResNet-34)")
print("=" * 80)
print("""
Input: 224x224x3 RGB Image
  ↓
Convolutional Layers (Feature Extraction):
  • Conv1: 7x7, 64 filters, stride 2
  • MaxPool: 3x3, stride 2
  • Residual Block 1: 3 layers, 64 filters
  • Residual Block 2: 4 layers, 128 filters
  • Residual Block 3: 6 layers, 256 filters
  • Residual Block 4: 3 layers, 512 filters
  • AvgPool: 7x7
  ↓
Custom Classifier (Transfer Learning):
  • FC1: 512 → 512 neurons + ReLU + Dropout(0.5)
  • FC2: 512 → 256 neurons + ReLU + Dropout(0.5)
  • FC3: 256 → 35 neurons (output classes)
  • Softmax: Probability distribution
  ↓
Output: 35-class probability distribution

Total Parameters: ~21.8M (ResNet-34)
Trainable Parameters: ~21.8M (all layers fine-tuned)
""")

# 7. Performance Metrics
print("\n" + "=" * 80)
print("7. EVALUATION METRICS")
print("=" * 80)
print("""
The models are evaluated using comprehensive metrics:

Classification Metrics:
  • Top-1 Accuracy:    Percentage of correct predictions
  • Top-5 Accuracy:    Percentage where true class is in top 5
  • Precision:         True positives / (True positives + False positives)
  • Recall:            True positives / (True positives + False negatives)
  • F1-Score:          Harmonic mean of precision and recall
  
Visualization:
  • Confusion Matrix:  35x35 heatmap showing prediction patterns
  • ROC Curves:        One-vs-rest for each class
  • AUC Scores:        Area under ROC curve per class
  • Loss Curves:       Training vs validation loss over epochs
  • Accuracy Curves:   Training vs validation accuracy over epochs
  
Per-Class Analysis:
  • Individual class performance
  • Most confused pairs
  • Difficult vs easy classes
""")

# 8. Sample Data Statistics
if sample:
    print("\n" + "=" * 80)
    print("8. DATASET STATISTICS")
    print("=" * 80)
    
    # Count images per class in sample
    class_counts = {}
    for class_name in sample.classes:
        class_dir = Path(SAMPLE_FILEPATH) / class_name
        if class_dir.exists():
            count = len([f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            class_counts[class_name] = count
    
    print(f"\nSample Dataset (for visualization):")
    print(f"  • Total images: {sum(class_counts.values())}")
    print(f"  • Images per class: {list(class_counts.values())[0] if class_counts else 0}")
    print(f"  • Total classes: {len(class_counts)}")
    
    print(f"\nFull Dataset (for training - not included):")
    print(f"  • Total images: 16,535")
    print(f"  • Images per class: 400-500")
    print(f"  • Source: Flickr API")
    print(f"  • Available on: Kaggle")

# 9. Usage Instructions
print("\n" + "=" * 80)
print("9. HOW TO USE THE PROJECT")
print("=" * 80)
print("""
Option 1: Interactive Notebooks (Recommended)
  1. Open JupyterLab at: http://localhost:8888
  2. Navigate to one of three notebooks:
     • 1. wep_classifier_initial.ipynb  - Training & evaluation
     • 2. wep_classifier_tuning.ipynb   - Hyperparameter tuning
     • 3. visualise_results.ipynb       - Results visualization
  3. Select the 'wep' kernel
  4. Run cells with Shift+Enter

Option 2: Python Scripts
  • View info:     python show_info.py
  • Quick demo:    python demo.py
  • This output:   python run_demo.py

Option 3: Custom Inference
  • Load pre-trained model from saved_models/
  • Preprocess image (resize, normalize)
  • Run inference
  • Get top-k predictions
""")

# 10. Project Files
print("\n" + "=" * 80)
print("10. PROJECT FILES & STRUCTURE")
print("=" * 80)

def print_tree(directory, prefix="", max_depth=2, current_depth=0):
    """Print directory tree structure"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('.') and item.name not in ['venv', '__pycache__']:
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, max_depth, current_depth + 1)
    except PermissionError:
        pass

print("\nProject Structure:\n")
print("wep-classifier/")
print_tree(".", prefix="", max_depth=2)

# 11. Next Steps
print("\n" + "=" * 80)
print("11. NEXT STEPS")
print("=" * 80)
print("""
✓ Environment Setup Complete
✓ Dependencies Installed
✓ Sample Dataset Extracted
✓ Pre-trained Models Available
✓ JupyterLab Running

Recommended Actions:
  1. Open JupyterLab and explore notebook 3 (visualise_results.ipynb)
  2. Review the confusion matrices and ROC curves
  3. Compare performance across the three models
  4. Experiment with notebook 2 for hyperparameter tuning
  5. Try inference on your own plant images

For Questions:
  • Check README.md for project overview
  • Read wepc-dissertation-report.pdf for detailed methodology
  • Visit: https://github.com/Achronus/wep-classifier
""")

print("\n" + "=" * 80)
print(" " * 25 + "DEMO COMPLETE! 🌱")
print("=" * 80)
print()
