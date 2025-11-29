"""
Flask Web Application for Wild Edible Plant Classifier
"""
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import base64
import numpy as np
from functions.model import Classifier

app = Flask(__name__)

# Global variables
model = None
LABELS = None
device = torch.device('cpu')

# Plant class names
PLANT_CLASSES = [
    'Alfalfa', 'Allium', 'Borage', 'Burdock', 'Calendula', 'Cattail',
    'Chickweed', 'Chicory', 'Chive Blossom', 'Coltsfoot', 'Common Mallow',
    'Common Milkweed', 'Common Vetch', 'Common Yarrow', 'Coneflower',
    'Cow Parsely', 'Cowslip', 'Crimson Clover', 'Crithmum Maritimum',
    'Daisy', 'Dandelion', 'Fennel', 'Firewood', 'Gardenia', 'Garlic Mustard',
    'Geranium', 'Ground Ivy', 'Harebell', 'Henbit', 'Knapweed',
    'Meadowsweet', 'Mullein', 'Pickerelweed', 'Ramsons', 'Red Clover'
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """Load the pre-trained ResNet-34 model"""
    global model, LABELS
    
    print("Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load('saved_models/best_resnet34.pt', 
                          map_location=device, 
                          weights_only=False)
    
    # Create model
    base_model = models.resnet34(weights=None)
    in_features = base_model.fc.in_features
    
    # Replace classifier
    base_model.fc = Classifier(
        in_features=in_features,
        out_features=35,
        hidden_layers=checkpoint['h_layers'],
        drop_prob=0.5
    )
    
    # Load weights
    base_model.load_state_dict(checkpoint['parameters'])
    base_model.eval()
    base_model.to(device)
    
    model = base_model
    LABELS = np.array(PLANT_CLASSES)
    
    print("✓ Model loaded successfully!")
    print(f"  Accuracy: {checkpoint['stats']['accuracy']*100:.2f}%")
    
    return checkpoint['stats']

def predict_image(image_bytes):
    """Make prediction on uploaded image"""
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.exp(output[0])  # Convert from log_softmax
        top_probs, top_indices = torch.topk(probabilities, 5)
    
    # Format results
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': LABELS[idx],
            'confidence': float(prob * 100)
        })
    
    return predictions

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        image_bytes = file.read()
        
        # Make prediction
        predictions = predict_image(image_bytes)
        
        # Convert image to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'image': f'data:image/jpeg;base64,{image_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    """Model information page"""
    stats = {
        'accuracy': 80.67,
        'top5_accuracy': 82.57,
        'precision': 77.14,
        'recall': 88.52,
        'f1_score': 82.44
    }
    return render_template('info.html', stats=stats, classes=PLANT_CLASSES)

if __name__ == '__main__':
    # Load model on startup
    stats = load_model()
    
    print("\n" + "=" * 60)
    print("🌱 Wild Edible Plant Classifier - Web Interface")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
