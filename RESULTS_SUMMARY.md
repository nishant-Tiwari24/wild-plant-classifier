# Wild Edible Plant Classifier - Results Summary 📊

## Project Output & Visualizations

This document summarizes the results and outputs from the Wild Edible Plant Classifier project.

---

## 🎯 Model Performance Overview

Three state-of-the-art CNN architectures were trained and evaluated:

### 1. **ResNet-34** (85.78 MB)
- Deep residual network with skip connections
- 34 layers with residual blocks
- Best for: High accuracy, complex feature learning

### 2. **GoogLeNet** (27.98 MB)
- Inception-based architecture
- Multi-scale feature extraction
- Best for: Balanced performance and efficiency

### 3. **MobileNet v2** (12.11 MB)
- Lightweight, mobile-optimized
- Depthwise separable convolutions
- Best for: Mobile deployment, fast inference

---

## 📈 Available Visualizations

The `plots/` directory contains comprehensive evaluation visualizations:

### Training Progress
- **`best_model_losses.png`** (34 KB)
  - Training vs validation loss curves
  - Shows model convergence over 20 epochs
  - Helps identify overfitting/underfitting

### Per-Model Results (for each architecture):

#### Confusion Matrices
- **`ResNet-34_cm.png`** (108 KB)
- **`GoogLeNet_cm.png`** (111 KB)
- **`MobileNet-V2_cm.png`** (108 KB)

**What they show:**
- 35×35 heatmap of predictions vs actual classes
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Identifies which plant species are confused with each other

#### Prediction Examples
- **`ResNet-34_preds.png`** (285 KB)
- **`GoogLeNet_preds.png`** (295 KB)
- **`MobileNet-V2_preds.png`** (299 KB)

**What they show:**
- Sample images with predicted vs actual labels
- Top-5 predictions with confidence scores
- Visual examples of correct and incorrect classifications
- Helps understand model behavior on real images

#### ROC Curves
- **`ResNet-34_roc.png`** (150 KB)
- **`GoogLeNet_roc.png`** (154 KB)
- **`MobileNet-V2_roc.png`** (146 KB)

**What they show:**
- Receiver Operating Characteristic curves
- One-vs-rest for each of the 35 classes
- AUC (Area Under Curve) scores
- Model's ability to distinguish between classes
- Higher AUC = better classification performance

---

## 📊 Evaluation Metrics

The models were evaluated using:

### Classification Metrics
- **Top-1 Accuracy**: Percentage of correct first predictions
- **Top-5 Accuracy**: Percentage where true class is in top 5 predictions
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall

### Per-Class Analysis
- Individual performance for each of 35 plant species
- Identification of difficult vs easy classes
- Most commonly confused plant pairs

### Visual Analysis
- Confusion matrices showing prediction patterns
- ROC curves with AUC scores per class
- Sample predictions with confidence scores
- Training/validation loss curves

---

## 🌿 Dataset Information

### Training Data
- **Total Images**: 16,535
- **Classes**: 35 wild edible plant species
- **Images per Class**: 400-500
- **Source**: Flickr API
- **Split**: 70% train / 15% validation / 15% test

### Data Augmentation
- Random rotation (±15°)
- Random horizontal flip
- Random crop (224×224)
- Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet statistics)

---

## 🔬 Training Configuration

### Hyperparameters
```
Epochs:          20
Learning Rate:   0.001
Batch Size:      64
Optimizer:       Adam
Loss Function:   Cross-Entropy
Weight Decay:    1e-4
```

### Transfer Learning
- **Base**: ImageNet pre-trained weights
- **Strategy**: Fine-tune all layers
- **Custom Classifier**: 
  - FC1: 512 neurons + ReLU + Dropout(0.5)
  - FC2: 256 neurons + ReLU + Dropout(0.5)
  - FC3: 35 neurons (output)

---

## 📁 Output Files

### Pre-trained Models (`saved_models/`)
```
best_resnet34.pt      - 85.78 MB - ResNet-34 weights
best_googlenet.pt     - 27.98 MB - GoogLeNet weights
best_mobilenetv2.pt   - 12.11 MB - MobileNet v2 weights
```

### Visualizations (`plots/`)
```
best_model_losses.png     - 34 KB  - Training curves
ResNet-34_cm.png          - 108 KB - Confusion matrix
ResNet-34_preds.png       - 285 KB - Prediction examples
ResNet-34_roc.png         - 150 KB - ROC curves
GoogLeNet_cm.png          - 111 KB - Confusion matrix
GoogLeNet_preds.png       - 295 KB - Prediction examples
GoogLeNet_roc.png         - 154 KB - ROC curves
MobileNet-V2_cm.png       - 108 KB - Confusion matrix
MobileNet-V2_preds.png    - 299 KB - Prediction examples
MobileNet-V2_roc.png      - 146 KB - ROC curves
```

---

## 🎓 Key Findings

### Model Comparison
The three architectures show different trade-offs:

**ResNet-34:**
- ✓ Highest accuracy
- ✓ Best feature learning
- ✗ Largest model size
- ✗ Slower inference

**GoogLeNet:**
- ✓ Balanced performance
- ✓ Multi-scale features
- ✓ Moderate size
- ✓ Good efficiency

**MobileNet v2:**
- ✓ Smallest model
- ✓ Fastest inference
- ✓ Mobile-friendly
- ✗ Slightly lower accuracy

### Challenging Classes
Some plant species are more difficult to classify due to:
- Visual similarity (e.g., different clover species)
- Seasonal variations in appearance
- Image quality and lighting conditions
- Overlapping features between species

### Success Factors
- Transfer learning from ImageNet
- Data augmentation for robustness
- Proper train/val/test split
- Dropout for regularization
- Fine-tuning all layers

---

## 🚀 How to View Results

### Option 1: JupyterLab (Interactive)
```bash
# JupyterLab is running at:
http://localhost:8888/lab?token=98ea27236e801de7bbd34771f5ed24797a61d7bdd9ef95b1

# Open: 3. visualise_results.ipynb
# Run all cells to regenerate plots
```

### Option 2: View Existing Plots
```bash
# Navigate to plots directory
cd wep-classifier/plots/

# View images with your default image viewer
open *.png  # macOS
```

### Option 3: Python Scripts
```bash
cd wep-classifier
source venv/bin/activate

# View project info
python show_info.py

# Run comprehensive demo
python run_demo.py
```

---

## 📚 Additional Resources

### Documentation
- **README.md** - Project overview
- **SETUP_COMPLETE.md** - Setup instructions
- **wepc-dissertation-report.pdf** - Full academic report

### Notebooks
1. **wep_classifier_initial.ipynb** - Initial training
2. **wep_classifier_tuning.ipynb** - Hyperparameter tuning
3. **visualise_results.ipynb** - Results visualization

### Code Structure
- **functions/model.py** - Classifier architecture
- **functions/plotting.py** - Visualization functions
- **functions/tuning.py** - Hyperparameter tuning
- **functions/utils.py** - Utility functions

---

## 🎯 Use Cases

This classifier can be used for:

1. **Educational Purposes**
   - Learning about wild edible plants
   - Understanding CNN architectures
   - Studying transfer learning

2. **Field Identification**
   - Mobile app for plant identification
   - Foraging assistance tool
   - Botanical education

3. **Research**
   - Comparing CNN architectures
   - Transfer learning experiments
   - Fine-tuning on custom datasets

4. **Production Deployment**
   - REST API for plant classification
   - Mobile app integration
   - Web-based identification tool

---

## ⚠️ Important Notes

### Limitations
- Model trained on Flickr images (may not generalize to all conditions)
- Requires good image quality for accurate predictions
- Should not be sole source for edibility determination
- Always verify with expert before consuming wild plants

### Safety Warning
**⚠️ CRITICAL: Never consume wild plants based solely on AI predictions!**
- Always consult expert botanists
- Use multiple identification methods
- Some plants are toxic and can be deadly
- This is an educational/research tool only

---

## 📞 Support & Resources

- **GitHub**: https://github.com/Achronus/wep-classifier
- **Dataset**: https://www.kaggle.com/ryanpartridge01/wild-edible-plants/
- **PyTorch**: https://pytorch.org/docs/

---

**Project Status**: ✅ Complete and Functional

All models trained, evaluated, and ready for use!
