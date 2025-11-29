#!/bin/bash

echo "=========================================="
echo "Wild Edible Plant Classifier - Web App"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask
fi

echo "Starting Flask server..."
echo ""
echo "🌱 Web interface will be available at:"
echo "   http://localhost:5000"
echo ""
echo "📝 Instructions:"
echo "   1. Open the URL in your browser"
echo "   2. Upload a plant image"
echo "   3. View AI predictions!"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the Flask app
python app.py
