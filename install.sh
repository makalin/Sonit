#!/bin/bash

# Sonit Installation Script
# Translating the Unspoken

set -e  # Exit on any error

echo "🎵 Sonit - Translating the Unspoken"
echo "=================================="
echo ""

# Check if Python 3.9+ is installed
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version found (>= $required_version required)"
else
    echo "❌ Python 3.9 or higher is required. Found: $python_version"
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip and try again."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p data/recordings
mkdir -p data/models
mkdir -p data/plots

# Set up database
echo ""
echo "Setting up database..."
python3 -c "
from utils.database import DatabaseManager
db = DatabaseManager()
print('✅ Database initialized')
"

# Make run script executable
chmod +x run.py

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To run Sonit:"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
echo "Or simply:"
echo "  ./run.py"
echo ""
echo "For more information, see README.md"
echo ""
echo "Happy translating! 🎵" 