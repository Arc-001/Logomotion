#!/bin/bash
# ============================================================================
# Manim Graph RAG Agent - Start Script
# ============================================================================
# Single point of entry to run the application
#
# Usage:
#   ./start.sh                    # Start API server (default)
#   ./start.sh index              # Index the dataset
#   ./start.sh generate "prompt"  # Generate a video
#   ./start.sh --docker           # Run via Docker
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Manim Graph RAG Agent                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for --docker flag
if [[ "$1" == "--docker" ]]; then
    echo "[DOCKER] Starting with Docker..."
    
    if ! command -v docker &> /dev/null; then
        echo "[ERROR] Docker is not installed."
        exit 1
    fi
    
    # Start services
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}[OK] Services started!${NC}"
    echo ""
    echo "API Server: http://localhost:8000"
    echo "Neo4j Browser: http://localhost:7474"
    echo ""
    echo "Commands:"
    echo "  Index dataset:  docker-compose exec app python -m src.main index --data /app/data"
    echo "  Generate video: docker-compose exec app python -m src.main generate -p 'your prompt'"
    echo "  View logs:      docker-compose logs -f"
    echo "  Stop:           docker-compose down"
    exit 0
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "[WARN] No .env file found. Creating from template..."
    cp .env.example .env
    echo "Please edit .env and add your API keys, then run again."
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

case "${1:-serve}" in
    "index")
        echo "Indexing dataset..."
        DATA_DIR="${2:-./extracted data}"
        python -m src.main index --data "$DATA_DIR"
        ;;
    
    "generate")
        if [ -z "$2" ]; then
            echo "Usage: ./start.sh generate \"your prompt here\" [-l length] [-d depth]"
            exit 1
        fi
        echo "Generating video..."
        # Pass prompt as first arg, then forward remaining args ($3 onwards)
        python -m src.main generate --prompt "$2" --output ./output "${@:3}"
        ;;
    
    "serve"|"api"|"server")
        echo "[SERVER] Starting API server..."
        echo ""
        echo -e "${GREEN}API available at: http://localhost:8000${NC}"
        echo "Press Ctrl+C to stop"
        echo ""
        python -m src.main serve --host 0.0.0.0 --port 8000
        ;;
    
    *)
        echo "Usage: ./start.sh [command]"
        echo ""
        echo "Commands:"
        echo "  serve              Start the API server (default)"
        echo "  index [path]       Index the Manim dataset"
        echo "  generate \"prompt\"  Generate a video from prompt"
        echo "  --docker           Run via Docker Compose"
        echo ""
        ;;
esac
