# Wild Edible Plant Classifier - Web Application Guide 🌱

## Overview

A beautiful, user-friendly web interface for the Wild Edible Plant Classifier. Upload plant images and get instant AI-powered predictions!

---

## 🚀 Quick Start

### Step 1: Start the Web Server

```bash
cd wep-classifier
source venv/bin/activate
python app.py
```

### Step 2: Open Your Browser

Navigate to: **http://localhost:5000**

### Step 3: Upload an Image

1. Click "Choose Image" or drag & drop a plant photo
2. Wait for the AI to analyze (2-3 seconds)
3. View the top 5 predictions with confidence scores!

---

## 📁 Project Structure

```
wep-classifier/
├── app.py                      # Flask backend server
├── templates/
│   ├── index.html             # Main classification page
│   └── info.html              # Model information page
├── static/
│   ├── css/
│   │   └── style.css          # Styling
│   └── js/
│       └── main.js            # Frontend logic
└── functions/
    └── model.py               # Model architecture
```

---

## 🎨 Features

### Main Page (/)
- **Drag & Drop Upload**: Simply drag an image onto the upload area
- **Click to Upload**: Traditional file selection
- **Real-time Preview**: See your uploaded image
- **Top-5 Predictions**: View the 5 most likely plant species
- **Confidence Scores**: Visual bars showing prediction confidence
- **Color-coded Results**: 
  - 🟢 Green (90-100%): Very High Confidence
  - 🔵 Blue (70-89%): High Confidence
  - 🟡 Yellow (50-69%): Medium Confidence
  - 🔴 Red (0-49%): Low Confidence

### Info Page (/info)
- **Model Performance**: Accuracy, precision, recall, F1-score
- **Architecture Details**: ResNet-34 specifications
- **Training Configuration**: Hyperparameters and dataset info
- **Plant Species List**: All 35 classifiable plants
- **How It Works**: Step-by-step explanation

---

## 🖼️ Testing the Application

### Option 1: Use Sample Images

The project includes sample images in `dataset/sample/`:

```bash
# Example: Test with a dandelion image
# Navigate to: dataset/sample/dandelion/
# Upload any image from that folder
```

### Option 2: Use Your Own Images

1. Take a photo of a plant with your phone
2. Transfer to your computer
3. Upload through the web interface

**Best Results:**
- Clear, well-lit photos
- Close-up of distinctive features (flowers, leaves)
- Single plant in focus
- Good image quality

---

## 📊 Understanding Results

### Prediction Format

```
🥇 #1  Dandelion          92.3%
🥈 #2  Daisy               3.2%
🥉 #3  Calendula           1.8%
   #4  Coneflower          1.1%
   #5  Common Yarrow       0.9%
```

### Confidence Levels

| Range | Level | Meaning | Action |
|-------|-------|---------|--------|
| 90-100% | ✅ Very High | Model is very confident | Likely correct |
| 70-89% | 👍 High | Model is confident | Probably correct |
| 50-69% | ⚠️ Medium | Model is uncertain | Verify with expert |
| 0-49% | ❌ Low | Model is guessing | Don't trust |

---

## 🔧 Technical Details

### Backend (Flask)

**Endpoints:**
- `GET /` - Main classification page
- `POST /predict` - Image upload and prediction
- `GET /info` - Model information page

**Request Format:**
```
POST /predict
Content-Type: multipart/form-data
Body: file (image file)
```

**Response Format:**
```json
{
  "success": true,
  "predictions": [
    {
      "class": "Dandelion",
      "confidence": 92.3
    },
    ...
  ],
  "image": "data:image/jpeg;base64,..."
}
```

### Frontend (HTML/CSS/JavaScript)

**Technologies:**
- Vanilla JavaScript (no frameworks)
- CSS Grid & Flexbox for layout
- Fetch API for AJAX requests
- Drag & Drop API for file upload

**Features:**
- Responsive design (mobile-friendly)
- Smooth animations
- Real-time feedback
- Error handling

---

## 🎯 Example Usage

### Example 1: Successful Prediction

**Input:** Photo of a dandelion flower

**Output:**
```
✅ Very High Confidence

🥇 #1  Dandelion          92.3%
🥈 #2  Daisy               3.2%
🥉 #3  Calendula           1.8%
   #4  Coneflower          1.1%
   #5  Common Yarrow       0.9%

The model is very confident about this prediction.
```

### Example 2: Uncertain Prediction

**Input:** Blurry photo of leaves

**Output:**
```
⚠️ Medium Confidence

🥇 #1  Common Mallow      45.2%
🥈 #2  Chickweed          38.1%
🥉 #3  Ground Ivy         12.3%
   #4  Geranium            3.1%
   #5  Borage              1.3%

The model is uncertain. Please verify with an expert.
```

---

## 🐛 Troubleshooting

### Server Won't Start

**Problem:** `Address already in use`

**Solution:**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use a different port
python app.py --port 5001
```

### Model Not Loading

**Problem:** `FileNotFoundError: saved_models/best_resnet34.pt`

**Solution:**
```bash
# Ensure you're in the correct directory
cd wep-classifier

# Check if model exists
ls -lh saved_models/
```

### Image Upload Fails

**Problem:** `Error: No file uploaded`

**Solution:**
- Ensure file is an image (JPG, PNG, etc.)
- Check file size (< 10MB recommended)
- Try a different browser

### Predictions Are Wrong

**Problem:** Low accuracy on test images

**Possible Causes:**
- Poor image quality
- Plant not in training set
- Unusual angle or lighting
- Multiple plants in image

**Solutions:**
- Use clear, well-lit photos
- Focus on distinctive features
- Try different angles
- Ensure single plant in frame

---

## 🔒 Security Notes

**Important:**
- This is a development server (Flask debug mode)
- Not suitable for production deployment
- No authentication or rate limiting
- File uploads not validated extensively

**For Production:**
- Use a production WSGI server (Gunicorn, uWSGI)
- Add authentication if needed
- Implement rate limiting
- Add file size/type validation
- Use HTTPS

---

## 🎨 Customization

### Change Colors

Edit `static/css/style.css`:

```css
/* Change gradient colors */
background: linear-gradient(135deg, #YOUR_COLOR_1 0%, #YOUR_COLOR_2 100%);
```

### Add More Plant Species

1. Retrain model with new classes
2. Update `PLANT_CLASSES` in `app.py`
3. Update model architecture if needed

### Modify Layout

Edit `templates/index.html` and `static/css/style.css`

---

## 📱 Mobile Support

The web app is fully responsive and works on:
- 📱 Smartphones (iOS, Android)
- 📱 Tablets
- 💻 Laptops
- 🖥️ Desktops

**Mobile Features:**
- Touch-friendly buttons
- Responsive grid layout
- Optimized image sizes
- Fast loading times

---

## 🚀 Deployment Options

### Option 1: Local Network

Share with devices on your network:

```bash
python app.py --host 0.0.0.0
# Access from other devices: http://YOUR_IP:5000
```

### Option 2: Cloud Deployment

**Platforms:**
- Heroku
- AWS Elastic Beanstalk
- Google Cloud Run
- Azure App Service
- DigitalOcean

**Requirements:**
- Add `requirements.txt`
- Configure production server
- Set environment variables
- Add domain/SSL

### Option 3: Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## 📊 Performance

**Metrics:**
- **Inference Time**: ~0.5-1 second per image (CPU)
- **Model Size**: 85.78 MB (ResNet-34)
- **Memory Usage**: ~500 MB RAM
- **Concurrent Users**: 1-5 (development server)

**Optimization Tips:**
- Use GPU for faster inference
- Implement caching
- Compress images before upload
- Use production WSGI server

---

## ⚠️ Important Warnings

### Safety Warning

**NEVER consume wild plants based solely on AI predictions!**

- Always consult expert botanists
- Use multiple identification methods
- Some plants are toxic and deadly
- This is an educational tool only

### Limitations

- Model trained on Flickr images
- May not generalize to all conditions
- Requires good image quality
- Limited to 35 plant species
- Not 100% accurate

---

## 📚 Additional Resources

- **Model Training**: See Jupyter notebooks
- **Dataset**: https://www.kaggle.com/ryanpartridge01/wild-edible-plants/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Flask Docs**: https://flask.palletsprojects.com/

---

## 🎉 Summary

You now have a fully functional web application for plant classification!

**What You Can Do:**
- ✅ Upload plant images
- ✅ Get instant predictions
- ✅ View confidence scores
- ✅ Learn about the model
- ✅ Test with sample images

**Next Steps:**
1. Start the server: `python app.py`
2. Open browser: http://localhost:5000
3. Upload a plant image
4. View predictions!

---

**Enjoy classifying plants! 🌱**
