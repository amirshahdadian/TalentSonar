#!/bin/bash
# Quick start script for JD2GH Streamlit app

echo "🚀 Starting JD2GH Streamlit App..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please copy .env.example to .env and add your API keys"
    echo ""
fi

# Run streamlit
echo "✅ Launching app at http://localhost:8501"
echo ""
streamlit run streamlit_app.py
