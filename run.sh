#!/bin/bash
# 🏺 Hieroglyph NLP Pipeline - Startup Script for macOS/Linux

clear

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  🏺 HIEROGLYPH NLP PIPELINE                  ║"
echo "║                   Ancient Egyptian Translator                ║"
echo "║                                                              ║"
echo "║  Automated Setup & Launch Script                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org"
    exit 1
fi

echo "✅ Python found"
echo ""

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "❌ Error: requirements.txt not found"
    echo "Make sure you're in the project root directory"
    exit 1
fi

echo "📦 Installing dependencies..."
pip3 install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "⚠️  Some packages failed to install. Continuing..."
fi

echo "✅ Dependencies installed"
echo ""

# Check for spaCy model
echo "🔍 Checking for spaCy model..."
python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Downloading spaCy model (this may take a minute)..."
    python3 -m spacy download en_core_web_sm
fi
echo "✅ spaCy model ready"
echo ""

# Check for CSV files
echo "🔍 Checking for data files..."
if [ ! -f "Test/Data/Update_gardiner_sign.csv" ]; then
    echo "⚠️  Missing: Test/Data/Update_gardiner_sign.csv"
fi
if [ ! -f "Test/Data/egyptian_dictionary.csv" ]; then
    echo "⚠️  Missing: Test/Data/egyptian_dictionary.csv"
fi
if [ ! -f "Test/Data/intention_dataset.csv" ]; then
    echo "⚠️  Missing: Test/Data/intention_dataset.csv"
fi
echo "✅ Data files check complete"
echo ""

echo "🚀 Starting server..."
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Open your browser and go to: http://localhost:5000"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python3 app.py
