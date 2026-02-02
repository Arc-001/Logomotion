#!/bin/bash
# ============================================================================
# Manim Graph RAG Agent - Installation Script
# ============================================================================
# Single point of entry to install all dependencies
#
# Usage:
#   ./install.sh           # Install with pip
#   ./install.sh --docker  # Build Docker images instead
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Manim Graph RAG Agent - Installation                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check for --docker flag
if [[ "$1" == "--docker" ]]; then
    echo "[DOCKER] Building Docker images..."
    echo ""
    
    # Check Docker is available
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Build the image
    docker-compose build
    
    echo ""
    echo "[OK] Docker images built successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Copy .env.example to .env and configure API keys"
    echo "  2. Run: ./start.sh --docker"
    exit 0
fi

# Regular pip installation
echo "[INSTALL] Installing Python dependencies..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env from template..."
    cp .env.example .env
    echo "[WARN] Please edit .env and add your API keys!"
fi

# Create output directories
mkdir -p output chroma_db

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Installation Complete!                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your OPENAI_API_KEY"
echo "  2. Start Neo4j: docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5-community"
echo "  3. Run: ./start.sh"
echo ""
