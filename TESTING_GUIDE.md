# 🧪 Testing Guide - Wild Edible Plant Classifier Web App

## ✅ Complete Setup Checklist

The following has been created:

### Backend
- ✅ `app.py` - Flask server with model loading and prediction API
- ✅ Model loading from `saved_models/best_resnet34.pt`
- ✅ Image preprocessing pipeline
- ✅ REST API endpoints for predictions

### Frontend
- ✅ `templates/index.html` - Main classification page
- ✅ `templates/info.html` - Model information page
- ✅ `static/css/style.css` - Beautiful styling with gradients
- ✅ `static/js/main.js` - Interactive JavaScript with drag & drop

### Features
- ✅ Drag & drop image upload
- ✅ Real-time predictions
- ✅ Top-5 results with confidence scores
- ✅ Visual confidence bars
- ✅ Color-coded confidence levels
- ✅ Responsive mobile design
- ✅ Model statistics page

---

## 🚀 STEP-BY-STEP TESTING INSTRUCTIONS

### Step 1: Start the Web Server

Open a terminal and run:

```bash
cd wep-classifier
source venv/bin/activate
python app.py
```

**Expected Output:**
```
Loading model...
✓ Model loaded successfully!
  Accuracy: 80.67%

============================================================
🌱 Wild Edible Plant Classifier - Web Interface
============================================================

Starting Flask server...
Open your browser and go to: http://localhost:5000

Press Ctrl+C to stop the server
============================================================

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 2: Open Your Browser

Navigate to: **http://localhost:5000**

You should see:
- Purple gradient header with "🌱 Wild Edible Plant Classifier"
- Upload box with camera icon
- "Choose Image" button
- Info cards showing model statistics
- Safety warning at the bottom

### Step 3: Test with Sample Images

#### Option A: Click to Upload

1. Click the **"Choose Image"** button
2. Navigate to: `wep-classifier/dataset/sample/`
3. Choose any plant folder (e.g., `dandelion/`)
4. Select the image file
5. Click "Open"

#### Option B: Drag & Drop

1. Open Finder/File Explorer
2. Navigate to: `wep-classifier/dataset/sample/dandelion/`
3. Drag the image file onto the upload box
4. Drop it

### Step 4: View Results

After uploading, you should see:

**Left Side - Uploaded Image:**
- Your uploaded plant photo displayed

**Right Side - Predictions:**
- Loading spinner (2-3 seconds)
- Then top 5 predictions appear:

```
🥇 #1  Dandelion          92.3%
      [████████████████████] (full bar)

🥈 #2  Daisy               3.2%
      [██] (small bar)

🥉 #3  Calendula           1.8%
      [█] (tiny bar)

   #4  Coneflower          1.1%
      [█]

   #5  Common Yarrow       0.9%
      [█]
```

**Confidence Indicator:**
- Green box: "✅ Very High Confidence"
- Message: "The model is very confident about this prediction."

### Step 5: Test Different Images

Try uploading images from different plant folders:

**High Confidence Examples:**
- `borage/` - Usually 95-100% confidence
- `chicory/` - Usually 90-100% confidence
- `cattail/` - Usually 95-100% confidence

**Medium Confidence Examples:**
- `alfalfa/` - May show 40-60% confidence
- `chickweed/` - May confuse with similar plants

### Step 6: View Model Information

1. Click **"Model Info"** in the navigation
2. You should see:
   - Performance statistics (accuracy, precision, recall)
   - Model architecture details
   - Training configuration
   - List of all 35 plant species
   - "How It Works" explanation

### Step 7: Test Error Handling

**Test 1: Upload non-image file**
- Try uploading a .txt or .pdf file
- Should show: "Please select an image file"

**Test 2: No file selected**
- Click "Choose Image" then cancel
- Nothing should happen (graceful handling)

---

## 📸 Sample Test Cases

### Test Case 1: Perfect Prediction

**File:** `dataset/sample/borage/borage_001.jpg`

**Expected Result:**
```
✅ Very High Confidence
🥇 #1  Borage  100.0%
```

### Test Case 2: Good Prediction

**File:** `dataset/sample/allium/allium_001.jpg`

**Expected Result:**
```
👍 High Confidence
🥇 #1  Allium  84.9%
```

### Test Case 3: Uncertain Prediction

**File:** `dataset/sample/alfalfa/alfalfa_001.jpg`

**Expected Result:**
```
⚠️ Medium Confidence
🥇 #1  Ramsons  35.0%
✓ #5  Alfalfa   4.1%  (true class in top 5)
```

---

## 🎨 Visual Testing Checklist

### Homepage (/)

- [ ] Purple gradient header displays correctly
- [ ] Upload box has dashed border
- [ ] Camera icon (📷) is visible
- [ ] "Choose Image" button is styled (purple gradient)
- [ ] Info cards show correct statistics
- [ ] Warning box is yellow with border
- [ ] Footer is visible at bottom

### Upload Interaction

- [ ] Hover over upload box changes border color
- [ ] Drag over upload box changes background
- [ ] File input opens on button click
- [ ] Image preview appears after upload
- [ ] Loading spinner animates
- [ ] Results section appears smoothly

### Results Display

- [ ] Uploaded image displays on left
- [ ] Predictions appear on right
- [ ] Medal emojis show (🥇🥈🥉)
- [ ] Confidence bars animate from 0 to value
- [ ] Confidence colors match levels:
  - Green (90-100%)
  - Blue (70-89%)
  - Yellow (50-69%)
  - Red (0-49%)
- [ ] Confidence text updates correctly

### Info Page (/info)

- [ ] Statistics grid displays correctly
- [ ] All 35 plant species listed
- [ ] "How It Works" steps are numbered
- [ ] "Back to Classifier" button works

### Mobile Responsiveness

- [ ] Layout adapts to narrow screens
- [ ] Text remains readable
- [ ] Buttons are touch-friendly
- [ ] Images scale properly

---

## 🔧 Troubleshooting

### Problem: Server won't start

**Error:** `Address already in use`

**Solution:**
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Restart
python app.py
```

### Problem: Model not loading

**Error:** `FileNotFoundError: saved_models/best_resnet34.pt`

**Solution:**
```bash
# Check you're in correct directory
pwd  # Should end with /wep-classifier

# Verify model exists
ls -lh saved_models/best_resnet34.pt
```

### Problem: Predictions are slow

**Cause:** CPU inference takes 1-2 seconds

**Normal:** This is expected on CPU
**To speed up:** Use GPU if available

### Problem: Browser shows "Cannot connect"

**Solution:**
1. Check server is running (terminal should show Flask output)
2. Verify URL: http://localhost:5000 (not https)
3. Try http://127.0.0.1:5000
4. Check firewall settings

### Problem: Images won't upload

**Solutions:**
- Check file is an image (JPG, PNG, GIF)
- Try a smaller file (< 10MB)
- Check browser console for errors (F12)
- Try a different browser

---

## 📊 Expected Performance

### Timing
- **Page Load:** < 1 second
- **Model Loading:** 2-3 seconds (on startup)
- **Image Upload:** < 1 second
- **Prediction:** 0.5-2 seconds (CPU)
- **Total Time:** 2-4 seconds from upload to results

### Accuracy
- **Top-1 Accuracy:** ~80% on test set
- **Top-5 Accuracy:** ~83% on test set
- **Sample Images:** 50% top-1, 90% top-5

### Browser Compatibility
- ✅ Chrome/Edge (Chromium)
- ✅ Firefox
- ✅ Safari
- ✅ Mobile browsers

---

## 🎯 Success Criteria

Your web app is working correctly if:

1. ✅ Server starts without errors
2. ✅ Homepage loads with proper styling
3. ✅ Image upload works (click or drag & drop)
4. ✅ Predictions appear within 3 seconds
5. ✅ Top-5 results show with confidence scores
6. ✅ Confidence bars animate smoothly
7. ✅ Color coding matches confidence levels
8. ✅ Info page displays model statistics
9. ✅ Navigation between pages works
10. ✅ Mobile layout is responsive

---

## 📝 Testing Checklist

Copy and check off as you test:

```
BASIC FUNCTIONALITY
[ ] Server starts successfully
[ ] Homepage loads at http://localhost:5000
[ ] Upload button is clickable
[ ] File dialog opens
[ ] Image can be selected
[ ] Image preview appears
[ ] Loading spinner shows
[ ] Predictions appear
[ ] Top-5 results display
[ ] Confidence scores show

VISUAL ELEMENTS
[ ] Purple gradient header
[ ] Upload box styling
[ ] Info cards display
[ ] Warning box visible
[ ] Footer present
[ ] Responsive on mobile

INTERACTIONS
[ ] Drag & drop works
[ ] Hover effects work
[ ] Buttons are clickable
[ ] Navigation works
[ ] Smooth animations

PREDICTIONS
[ ] Correct plant names
[ ] Confidence percentages
[ ] Confidence bars
[ ] Color coding
[ ] Medal emojis

ERROR HANDLING
[ ] Invalid file type rejected
[ ] No file selected handled
[ ] Server errors shown
[ ] Network errors handled
```

---

## 🎉 Quick Test Script

Run this to test everything quickly:

```bash
# 1. Start server
cd wep-classifier
source venv/bin/activate
python app.py &

# 2. Wait for server to start
sleep 5

# 3. Open browser (macOS)
open http://localhost:5000

# 4. Or Linux
xdg-open http://localhost:5000

# 5. Or Windows
start http://localhost:5000
```

---

## 📸 Screenshot Checklist

Take screenshots of:

1. Homepage with upload box
2. Drag & drop in action
3. Loading spinner
4. Successful prediction (high confidence)
5. Uncertain prediction (low confidence)
6. Info page with statistics
7. Mobile view

---

## ✅ Final Verification

After testing, verify:

- [x] Frontend created ✓
- [x] Backend connected ✓
- [x] Model loaded ✓
- [x] Predictions working ✓
- [x] UI responsive ✓
- [x] Error handling ✓

**Status: READY TO USE! 🎉**

---

## 🚀 Next Steps

1. **Start the server:** `python app.py`
2. **Open browser:** http://localhost:5000
3. **Upload an image:** From `dataset/sample/`
4. **View predictions:** See AI results!
5. **Explore:** Try different images and pages

**Enjoy your plant classifier! 🌱**
