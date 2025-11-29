# Wild Edible Plant Classifier - Setup Complete! 🎉

## Project Successfully Cloned and Running

The **Wild Edible Plant Classifier** project has been successfully cloned from GitHub and is now running on your system.

---

## 📊 Project Summary

This is a deep learning project that classifies **35 species of wild edible plants** using transfer learning with three state-of-the-art CNN architectures:

- **MobileNet v2** (12.11 MB) - Lightweight, mobile-optimized
- **GoogLeNet** (27.98 MB) - Inception-based architecture  
- **ResNet-34** (85.78 MB) - Residual network with skip connections

### Dataset
- **16,535 images** total
- **35 plant species** (400-500 images per class)
- Images collected from Flickr API
- Sample dataset extracted and ready to use

---

## 🚀 Access JupyterLab

JupyterLab is currently running and accessible at:

**URL:** http://localhost:8888/lab?token=98ea27236e801de7bbd34771f5ed24797a61d7bdd9ef95b1

### How to Use:
1. Click the URL above or copy it to your browser
2. JupyterLab will open automatically
3. Navigate to one of the three notebooks:
   - `1. wep_classifier_initial.ipynb` - Initial training
   - `2. wep_classifier_tuning.ipynb` - Hyperparameter tuning
   - `3. visualise_results.ipynb` - Results visualization
4. Select the **"wep"** kernel when prompted
5. Run cells using **Shift+Enter**

---

## 🌿 Plant Species Classified

The model can identify these 35 wild edible plants:

1. Alfalfa
2. Allium
3. Borage
4. Burdock
5. Calendula
6. Cattail
7. Chickweed
8. Chicory
9. Chive Blossom
10. Coltsfoot
11. Common Mallow
12. Common Milkweed
13. Common Vetch
14. Common Yarrow
15. Coneflower
16. Cow Parsely
17. Cowslip
18. Crimson Clover
19. Crithmum Maritimum
20. Daisy
21. Dandelion
22. Fennel
23. Firewood
24. Gardenia
25. Garlic Mustard
26. Geranium
27. Ground Ivy
28. Harebell
29. Henbit
30. Knapweed
31. Meadowsweet
32. Mullein
33. Pickerelweed
34. Ramsons
35. Red Clover

---

## 🤖 Pre-trained Models Available

All three pre-trained models are included and ready to use:

- ✅ **ResNet-34** (85.78 MB)
- ✅ **GoogLeNet** (27.98 MB)  
- ✅ **MobileNet v2** (12.11 MB)

These models were trained on the full dataset and can be used for:
- Inference on new images
- Transfer learning
- Fine-tuning on custom datasets
- Performance comparison

---

## ⚙️ Training Configuration

The models were trained with:

- **Epochs:** 20
- **Learning Rate:** 0.001
- **Batch Size:** 64
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy
- **Data Split:** 70% train / 15% validation / 15% test
- **Data Augmentation:** Random rotation, flip, crop
- **Transfer Learning:** Pre-trained ImageNet weights

---

## 📈 Evaluation Metrics

The notebooks include comprehensive evaluation:

- **Accuracy** (Top-1 and Top-5)
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC Curves and AUC**
- **Training/Validation Loss Curves**
- **Per-class Performance Analysis**

---

## 📁 Project Structure

```
wep-classifier/
├── dataset/
│   └── sample/                    # Sample images (1 per class)
├── functions/
│   ├── model.py                   # Classifier architecture
│   ├── plotting.py                # Visualization functions
│   ├── tuning.py                  # Hyperparameter tuning
│   └── utils.py                   # Utility functions
├── saved_models/                  # Pre-trained model weights
│   ├── best_resnet34.pt
│   ├── best_googlenet.pt
│   └── best_mobilenetv2.pt
├── plots/                         # Generated visualizations
├── 1. wep_classifier_initial.ipynb
├── 2. wep_classifier_tuning.ipynb
├── 3. visualise_results.ipynb
├── demo.py                        # Quick demo script
├── show_info.py                   # Project information
└── README.md                      # Original project README
```

---

## 🔧 Environment Details

- **Python:** 3.9.6
- **PyTorch:** 2.8.0
- **TorchVision:** 0.23.0
- **JupyterLab:** 4.5.0
- **Device:** CPU (MPS backend available on Apple Silicon)
- **Virtual Environment:** `venv/` (activated)
- **IPython Kernel:** "wep" (installed)

---

## 📝 Quick Commands

### View Project Information
```bash
cd wep-classifier
source venv/bin/activate
python show_info.py
```

### Run Demo
```bash
cd wep-classifier
source venv/bin/activate
python demo.py
```

### Stop JupyterLab
Press `Ctrl+C` twice in the terminal where JupyterLab is running

### Restart JupyterLab
```bash
cd wep-classifier
source venv/bin/activate
jupyter-lab --no-browser
```

---

## 📚 Additional Resources

- **Original Repository:** https://github.com/Achronus/wep-classifier
- **Kaggle Dataset:** https://www.kaggle.com/ryanpartridge01/wild-edible-plants/
- **PyTorch Documentation:** https://pytorch.org/docs/
- **Dissertation Report:** `wepc-dissertation-report.pdf` (included)

---

## ✨ What's Next?

1. **Explore the Notebooks** - Start with notebook 1 to see the training process
2. **Visualize Results** - Check notebook 3 for performance comparisons
3. **Experiment** - Try different hyperparameters in notebook 2
4. **Custom Dataset** - Adapt the code for your own plant classification task
5. **Deploy** - Use the pre-trained models for inference in production

---

## 🎓 Academic Context

This project was created as part of a BSc dissertation focusing on:
- Transfer Learning for image classification
- Comparison of CNN architectures
- Wild edible plant identification
- Deep learning best practices

The full dissertation report is available in `wepc-dissertation-report.pdf`.

---

**Enjoy exploring the Wild Edible Plant Classifier! 🌱**
