#!/usr/bin/env python3
"""
Demo script for Wild Edible Plant Classifier
"""
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets
from pathlib import Path

print("=" * 60)
print("Wild Edible Plant Classifier - Demo")
print("=" * 60)

# Check PyTorch and CUDA availability
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("Device: CPU")

# Load sample dataset
SAMPLE_FILEPATH = 'dataset/sample'
SAMPLE_TRANSFORM = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print(f"\n{'=' * 60}")
print("Loading Sample Dataset...")
print("=" * 60)

sample = torchvision.datasets.ImageFolder(SAMPLE_FILEPATH, transform=SAMPLE_TRANSFORM)
print(f"\nDataset loaded successfully!")
print(f"Number of classes: {len(sample.classes)}")
print(f"Number of samples: {len(sample)}")

# Display class names
print(f"\n{'=' * 60}")
print("Plant Classes:")
print("=" * 60)
for idx, class_name in enumerate(sample.classes, 1):
    print(f"{idx:2d}. {class_name}")

# Check for saved models
print(f"\n{'=' * 60}")
print("Checking for Pre-trained Models...")
print("=" * 60)

models_dir = Path('saved_models')
if models_dir.exists():
    models = list(models_dir.glob('*.pt'))
    if models:
        print(f"\nFound {len(models)} pre-trained model(s):")
        for model in models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"  - {model.name} ({size_mb:.2f} MB)")
    else:
        print("\nNo pre-trained models found.")
else:
    print("\nModels directory not found.")

# Display project structure
print(f"\n{'=' * 60}")
print("Project Structure:")
print("=" * 60)
print("""
This project contains:
1. Three Jupyter Notebooks:
   - 1. wep_classifier_initial.ipynb (Initial training)
   - 2. wep_classifier_tuning.ipynb (Model tuning)
   - 3. visualise_results.ipynb (Results visualization)

2. Three CNN Architectures:
   - MobileNet v2
   - GoogLeNet
   - ResNet-34

3. Dataset:
   - 35 classes of wild edible plants
   - 16,535 images total (400-500 per class)
   - Images sourced from Flickr API
""")

print(f"\n{'=' * 60}")
print("Setup Complete!")
print("=" * 60)
print("\nTo run the notebooks:")
print("1. Open JupyterLab at: http://localhost:8888")
print("2. Use the token from the terminal output")
print("3. Open any of the three notebooks")
print("4. Select the 'wep' kernel")
print("5. Run the cells!")
print("\n" + "=" * 60)
