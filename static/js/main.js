// Main JavaScript for Wild Edible Plant Classifier

const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const resultsSection = document.getElementById('resultsSection');
const previewImage = document.getElementById('previewImage');
const loading = document.getElementById('loading');
const predictionResults = document.getElementById('predictionResults');

// Handle file selection
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop functionality
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = 'white';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = 'white';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
});

function handleFileSelect() {
    const file = fileInput.files[0];
    
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, etc.)');
        return;
    }
    
    // Show results section
    resultsSection.style.display = 'grid';
    loading.style.display = 'block';
    predictionResults.style.display = 'none';
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Upload and predict
    uploadAndPredict(file);
}

async function uploadAndPredict(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictions(data.predictions);
        } else {
            showError(data.error || 'An error occurred during prediction');
        }
    } catch (error) {
        showError('Failed to connect to server: ' + error.message);
    }
}

function displayPredictions(predictions) {
    loading.style.display = 'none';
    predictionResults.style.display = 'block';
    
    // Clear previous results
    predictionResults.innerHTML = '';
    
    // Add predictions
    predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const medal = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '';
        
        item.innerHTML = `
            <div class="prediction-header">
                <span class="prediction-rank">${medal} #${index + 1}</span>
                <span class="prediction-class">${pred.class}</span>
                <span class="prediction-confidence">${pred.confidence.toFixed(1)}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${pred.confidence}%"></div>
            </div>
        `;
        
        predictionResults.appendChild(item);
        
        // Animate confidence bar
        setTimeout(() => {
            const fill = item.querySelector('.confidence-fill');
            fill.style.width = pred.confidence + '%';
        }, 100 * index);
    });
    
    // Add interpretation
    const topPrediction = predictions[0];
    const interpretation = document.createElement('div');
    interpretation.style.marginTop = '20px';
    interpretation.style.padding = '15px';
    interpretation.style.borderRadius = '10px';
    interpretation.style.background = getConfidenceColor(topPrediction.confidence);
    interpretation.style.color = 'white';
    interpretation.style.fontWeight = 'bold';
    interpretation.innerHTML = `
        <div style="font-size: 1.2em; margin-bottom: 5px;">
            ${getConfidenceEmoji(topPrediction.confidence)} ${getConfidenceText(topPrediction.confidence)}
        </div>
        <div style="font-size: 0.9em; opacity: 0.9;">
            ${getConfidenceAdvice(topPrediction.confidence)}
        </div>
    `;
    
    predictionResults.appendChild(interpretation);
}

function getConfidenceColor(confidence) {
    if (confidence >= 90) return 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
    if (confidence >= 70) return 'linear-gradient(135deg, #17a2b8 0%, #138496 100%)';
    if (confidence >= 50) return 'linear-gradient(135deg, #ffc107 0%, #e0a800 100%)';
    return 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
}

function getConfidenceEmoji(confidence) {
    if (confidence >= 90) return '✅';
    if (confidence >= 70) return '👍';
    if (confidence >= 50) return '⚠️';
    return '❌';
}

function getConfidenceText(confidence) {
    if (confidence >= 90) return 'Very High Confidence';
    if (confidence >= 70) return 'High Confidence';
    if (confidence >= 50) return 'Medium Confidence';
    return 'Low Confidence';
}

function getConfidenceAdvice(confidence) {
    if (confidence >= 90) return 'The model is very confident about this prediction.';
    if (confidence >= 70) return 'The model is fairly confident. Consider verifying with other sources.';
    if (confidence >= 50) return 'The model is uncertain. Please verify with an expert.';
    return 'The model has low confidence. Do not trust this prediction.';
}

function showError(message) {
    loading.style.display = 'none';
    predictionResults.style.display = 'block';
    predictionResults.innerHTML = `
        <div style="background: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border: 2px solid #f5c6cb;">
            <strong>❌ Error:</strong><br>
            ${message}
        </div>
    `;
}

// Add keyboard shortcut for file selection
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'o') {
        e.preventDefault();
        fileInput.click();
    }
});
