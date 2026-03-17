@echo off
REM 🏺 Hieroglyph NLP Pipeline - Startup Script for Windows

cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                  🏺 HIEROGLYPH NLP PIPELINE                  ║
echo ║                   Ancient Egyptian Translator                ║
echo ║                                                              ║
echo ║  Automated Setup & Launch Script                            ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if requirements.txt exists
if not exist requirements.txt (
    echo ❌ Error: requirements.txt not found
    echo Make sure you're in the project root directory
    pause
    exit /b 1
)

echo 📦 Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ⚠️  Some packages failed to install. Continuing...
)

echo ✅ Dependencies installed
echo.

REM Check for spaCy model
echo 🔍 Checking for spaCy model...
python -c "import spacy; spacy.load('en_core_web_sm')" >nul 2>&1
if errorlevel 1 (
    echo 📥 Downloading spaCy model (this may take a minute)...
    python -m spacy download en_core_web_sm
)
echo ✅ spaCy model ready
echo.

REM Check for CSV files
echo 🔍 Checking for data files...
if not exist "Test\Data\Update_gardiner_sign.csv" (
    echo ⚠️  Missing: Test\Data\Update_gardiner_sign.csv
)
if not exist "Test\Data\egyptian_dictionary.csv" (
    echo ⚠️  Missing: Test\Data\egyptian_dictionary.csv
)
if not exist "Test\Data\intention_dataset.csv" (
    echo ⚠️  Missing: Test\Data\intention_dataset.csv
)
echo ✅ Data files check complete
echo.

echo 🚀 Starting server...
echo.
echo ═══════════════════════════════════════════════════════════════
echo Open your browser and go to: http://localhost:5000
echo ═══════════════════════════════════════════════════════════════
echo.

python app.py

pause
