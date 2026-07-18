#!/bin/bash
# ============================================================================
# Logomotion - Single Entry Point
# ============================================================================
# One script for the whole lifecycle:
#
#   ./start.sh install [--docker]   Install deps (venv + pip, or build images)
#   ./start.sh start [--docker]     Start Neo4j (+ seed) and the API server
#   ./start.sh stop                 Stop all containers
#   ./start.sh seed [--force]       Load the bundled Neo4j dump (--force re-seeds)
#   ./start.sh serve                Run the API server only (local venv)
#   ./start.sh index [path]         Index the Manim dataset
#   ./start.sh generate "prompt"    Generate a video
#   ./start.sh test                 Run the test suite
#   ./start.sh logs                 Follow container logs
#   ./start.sh status               Show container status
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

banner() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                        Logomotion                            ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# docker compose v2 with v1 fallback
compose() {
    if docker compose version &> /dev/null; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

require_docker() {
    if ! command -v docker &> /dev/null; then
        echo "[ERROR] Docker is not installed."
        exit 1
    fi
}

ensure_env() {
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}[WARN] No .env file found. Creating from template...${NC}"
        cp .env.example .env
        echo "Please edit .env and add your API keys, then run again."
        exit 1
    fi
    set -a
    source .env
    set +a
}

activate_venv() {
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    export PATH="$HOME/.local/bin:$PATH"
}

cmd_install() {
    banner
    if [[ "$1" == "--docker" ]]; then
        require_docker
        echo "[INSTALL] Building Docker images..."
        compose build
        echo ""
        echo -e "${GREEN}[OK] Docker images built.${NC}"
        echo "Next: edit .env, then ./start.sh start --docker"
        return
    fi

    echo "[INSTALL] Installing Python dependencies..."
    python3 --version

    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    if [ ! -f ".env" ]; then
        echo "Creating .env from template..."
        cp .env.example .env
        echo -e "${YELLOW}[WARN] Edit .env and add your OPENROUTER_API_KEY.${NC}"
    fi

    mkdir -p output chroma_db

    echo ""
    echo -e "${GREEN}[OK] Installation complete.${NC}"
    echo "Next: edit .env, then ./start.sh start"
}

cmd_seed() {
    banner
    require_docker
    if [[ "$1" == "--force" ]]; then
        echo "[SEED] Forcing re-seed: stopping Neo4j and clearing the seed marker..."
        compose stop neo4j 2> /dev/null || true
        compose run --rm --entrypoint bash neo4j-seed -c "rm -f /data/.seeded"
    fi
    # neo4j-admin load needs the database offline
    compose stop neo4j 2> /dev/null || true
    compose up neo4j-seed
    compose up -d neo4j
    echo ""
    echo -e "${GREEN}[OK] Neo4j is up (seeded knowledge base).${NC}"
}

cmd_start() {
    banner
    if [[ "$1" == "--docker" ]]; then
        require_docker
        echo "[START] Starting full stack with Docker..."
        compose up -d
        echo ""
        echo -e "${GREEN}[OK] Services started.${NC}"
        echo ""
        echo "API + Web UI:  http://localhost:8000"
        echo "Neo4j Browser: http://localhost:7474"
        echo ""
        echo "Logs: ./start.sh logs    Stop: ./start.sh stop"
        return
    fi

    # Local mode: containers for the databases, venv for the app
    require_docker
    ensure_env
    activate_venv
    echo "[START] Bringing up Neo4j (with seed) ..."
    compose up -d neo4j
    echo ""
    echo -e "${GREEN}API + Web UI at: http://localhost:8000${NC}"
    echo "Press Ctrl+C to stop the server (containers keep running; ./start.sh stop)"
    echo ""
    python -m src.main serve --host 0.0.0.0 --port 8000
}

cmd_stop() {
    banner
    require_docker
    echo "[STOP] Stopping containers..."
    compose down
    echo -e "${GREEN}[OK] Stopped. Data volumes are preserved.${NC}"
}

usage() {
    echo "Usage: ./start.sh <command>"
    echo ""
    echo "Lifecycle:"
    echo "  install [--docker]   Install deps (venv + pip, or build Docker images)"
    echo "  start [--docker]     Start Neo4j (+ seed) and the API server"
    echo "  stop                 Stop all containers (volumes preserved)"
    echo "  seed [--force]       Load the bundled Neo4j dump (--force re-seeds)"
    echo ""
    echo "Everything else:"
    echo "  serve                Run the API server only (local venv)"
    echo "  index [path]         Index the Manim dataset"
    echo "  generate \"prompt\"    Generate a video (extra flags forwarded)"
    echo "  test                 Run the test suite"
    echo "  logs                 Follow container logs"
    echo "  status               Show container status"
}

case "${1:-help}" in
    "install"|"setup")
        cmd_install "$2"
        ;;

    "start")
        cmd_start "$2"
        ;;

    "stop"|"down")
        cmd_stop
        ;;

    "seed")
        cmd_seed "$2"
        ;;

    "index")
        banner
        ensure_env
        activate_venv
        echo "Indexing dataset..."
        DATA_DIR="${2:-./extracted data}"
        python -m src.main index --data "$DATA_DIR"
        ;;

    "generate")
        banner
        if [ -z "$2" ]; then
            echo "Usage: ./start.sh generate \"your prompt here\" [-l length] [-d depth] [-q quality] [--visual-qa]"
            exit 1
        fi
        ensure_env
        activate_venv
        echo "Generating video..."
        python -m src.main generate --prompt "$2" --output ./output "${@:3}"
        ;;

    "serve"|"api"|"server")
        banner
        ensure_env
        activate_venv
        echo "[SERVER] Starting API server..."
        echo ""
        echo -e "${GREEN}API + Web UI at: http://localhost:8000${NC}"
        echo "Press Ctrl+C to stop"
        echo ""
        python -m src.main serve --host 0.0.0.0 --port 8000
        ;;

    "test"|"tests")
        banner
        activate_venv
        python -m pytest tests/ "${@:2}"
        ;;

    "logs")
        require_docker
        compose logs -f
        ;;

    "status"|"ps")
        require_docker
        compose ps
        ;;

    "--docker")
        # Backward compatibility: ./start.sh --docker == ./start.sh start --docker
        cmd_start "--docker"
        ;;

    "help"|"-h"|"--help"|*)
        banner
        usage
        ;;
esac
