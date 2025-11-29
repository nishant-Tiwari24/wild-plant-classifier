#!/usr/bin/env python3
"""
Display information about the Wild Edible Plant Classifier project
"""
import torch
import os
from pathlib import Path

print("=" * 80)
print(" " * 20 + "WILD EDIBLE PLANT CLASSIFIER")
print("=" * 80)

# System Information
print("\n📊 SYSTEM INFORMATION")
print("─" * 80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("Device: CPU (MPS backend available on Apple Silicon)")

# Project Overview
print("\n📚 PROJECT OVERVIEW")
print("─" * 80)
print("""
This project implements a deep learning classifier for identifying 35 species of
wild edible plants using transfer learning with three state-of-the-art CNN 
architectures:

  • MobileNet v2  - Lightweight, mobile-optimized architecture
  • GoogLeNet     - Inception-based architecture with multiple scales
  • ResNet-34     - Residual network with skip connections

The models were trained on 16,535 images collected from Flickr API, with 400-500
images per plant species.
""")

# Dataset Information
print("\n🌿 PLANT SPECIES (35 classes)")
print("─" * 80)

plants = [
    "Alfalfa", "Allium", "Borage", "Burdock", "Calendula", "Cattail",
    "Chickweed", "Chicory", "Chive Blossom", "Coltsfoot", "Common Mallow",
    "Common Milkweed", "Common Vetch", "Common Yarrow", "Coneflower",
    "Cow Parsely", "Cowslip", "Crimson Clover", "Crithmum Maritimum",
    "Daisy", "Dandelion", "Fennel", "Firewood", "Gardenia", "Garlic Mustard",
    "Geranium", "Ground Ivy", "Harebell", "Henbit", "Knapweed",
    "Meadowsweet", "Mullein", "Pickerelweed", "Ramsons", "Red Clover"
]

# Print in 3 columns
for i in range(0, len(plants), 3):
    row = plants[i:i+3]
    print(f"  {i+1:2d}. {row[0]:20s}", end="")
    if len(row) > 1:
        print(f"  {i+2:2d}. {row[1]:20s}", end="")
    if len(row) > 2:
        print(f"  {i+3:2d}. {row[2]:20s}")
    else:
        print()

# Model Information
print("\n🤖 PRE-TRAINED MODELS")
print("─" * 80)

models_dir = Path('saved_models')
if models_dir.exists():
    models = {
        'best_resnet34.pt': 'ResNet-34',
        'best_googlenet.pt': 'GoogLeNet',
        'best_mobilenetv2.pt': 'MobileNet v2'
    }
    
    for model_file, model_name in models.items():
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {model_name:15s} - {size_mb:6.2f} MB - {model_file}")
        else:
            print(f"  ✗ {model_name:15s} - Not found")
else:
    print("  ⚠ Models directory not found")

# Notebooks
print("\n📓 JUPYTER NOTEBOOKS")
print("─" * 80)
notebooks = [
    ("1. wep_classifier_initial.ipynb", "Initial model training and evaluation"),
    ("2. wep_classifier_tuning.ipynb", "Hyperparameter tuning and optimization"),
    ("3. visualise_results.ipynb", "Results visualization and comparison")
]

for nb_file, description in notebooks:
    exists = "✓" if os.path.exists(nb_file) else "✗"
    print(f"  {exists} {nb_file:35s} - {description}")

# Training Details
print("\n⚙️  TRAINING CONFIGURATION")
print("─" * 80)
print("""
  • Epochs: 20
  • Learning Rate: 0.001
  • Batch Size: 64
  • Train/Val/Test Split: 70% / 15% / 15%
  • Optimizer: Adam
  • Loss Function: Cross-Entropy
  • Data Augmentation: Random rotation, flip, crop
  • Transfer Learning: Pre-trained ImageNet weights
""")

# Performance Metrics
print("\n📈 EVALUATION METRICS")
print("─" * 80)
print("""
The models are evaluated using:
  • Accuracy (Top-1 and Top-5)
  • Precision, Recall, F1-Score
  • Confusion Matrix
  • ROC Curves and AUC
  • Training/Validation Loss Curves
""")

# How to Use
print("\n🚀 HOW TO USE")
print("─" * 80)
print("""
1. JupyterLab is running at: http://localhost:8888
   
2. Copy the token from the terminal output when you started JupyterLab

3. Open any of the three notebooks:
   • Start with notebook 1 for initial training
   • Use notebook 2 for hyperparameter tuning
   • View notebook 3 for results visualization

4. Select the 'wep' kernel when prompted

5. Run cells sequentially using Shift+Enter

Note: Training from scratch requires the full dataset (not included in sample).
      The pre-trained models can be used for inference immediately.
""")

# File Structure
print("\n📁 PROJECT STRUCTURE")
print("─" * 80)
print("""
wep-classifier/
├── dataset/
│   └── sample/              # Sample images (1 per class)
├── functions/
│   ├── model.py            # Classifier architecture
│   ├── plotting.py         # Visualization functions
│   ├── tuning.py           # Hyperparameter tuning
│   └── utils.py            # Utility functions
├── saved_models/           # Pre-trained model weights
├── plots/                  # Generated visualizations
├── 1. wep_classifier_initial.ipynb
├── 2. wep_classifier_tuning.ipynb
└── 3. visualise_results.ipynb
""")

print("=" * 80)
print(" " * 25 + "Setup Complete! 🎉")
print("=" * 80)
print()
