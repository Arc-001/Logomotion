# Logomotion

Logomotion is an agentic system that turns a natural-language prompt into a narrated mathematical or technical animation. It combines Graph RAG retrieval over a knowledge base of Manim examples with a self-correcting LangGraph workflow: it plans the video, generates Manim code, renders it, reviews the rendered frames, synthesizes a voiceover locally, and merges everything into a final video.

## Demos

- [Demo 1](https://github.com/Arc-001/Logomotion/raw/main/assets/1.mp4)
- [Demo 2](https://github.com/Arc-001/Logomotion/raw/main/assets/2.mp4)
- [Demo 3](https://github.com/Arc-001/Logomotion/raw/main/assets/3.mp4)

## Architecture

![System Architecture](assets/architecture.png)

The pipeline is a LangGraph state machine with parallel audio and video branches joined by a barrier before the final merge.

| Stage | Role |
|---|---|
| `storyboard` | Plans the video as timed sections (title, duration, visuals, narration) before any code is written. Toggle with `STORYBOARD_ENABLED` (default on). |
| `video_code_gen` | Retrieves similar Manim examples via hybrid search (ChromaDB vectors plus Neo4j graph traversal, guided by class and animation hints mined from the prompt), builds a constrained prompt, and calls the LLM to produce scene code and a timestamped transcript. |
| `code_executor` | Statically validates the code (syntax, scene class, removed APIs), then renders it with `manim render` at the configured quality. |
| `recorrector` | On render failure, feeds the error and code back to the LLM for a fix. Loops up to `MAX_RETRIES` times (default 3). |
| `visual_qa` | Optional. Samples frames from the rendered video and asks the multimodal LLM to find layout problems (overlaps, off-frame content, empty screens). A layout-only corrector fixes the code and re-renders, capped by `VISUAL_QA_MAX_ATTEMPTS`. |
| `transcript_processor` | Runs in parallel with the code branch. Splits the transcript into segments and synthesizes each with Kokoro TTS (an 82M-parameter model that runs locally on CPU). |
| `render_checker` | Validates the rendered video: integrity, resolution, codec, and duration against the target. |
| `synchronizer` | Joins the audio and video branches and rescales transcript timestamps if the video was time-adjusted. |
| `audio_video_merger` | Builds the narration track (silence padding, tempo adjustment), merges it with the video via ffmpeg, persists the result to `output/`, and cleans up all per-job temporary files. |

## Requirements

- Python 3.10 or newer
- System packages for Manim: `libcairo2-dev`, `libpango1.0-dev`, `ffmpeg`, and a LaTeX distribution (see the `Dockerfile` for the full list)
- An OpenRouter API key, or any OpenAI-compatible endpoint
- Neo4j 5 or newer (bundled via Docker Compose)

## Installation

```bash
git clone https://github.com/Arc-001/Logomotion.git
cd Logomotion
./install.sh
```

This creates a virtual environment, installs all Python dependencies (including Manim), and sets up the output directories. Copy `.env.example` to `.env` and fill in your API key before running.

For a fully containerized setup:

```bash
./install.sh --docker
```

## Configuration

All configuration lives in `.env` (see `.env.example` for a complete template).

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | API key for the LLM provider | required |
| `OPENROUTER_MODEL` | Model identifier | `google/gemini-3-flash-preview` |
| `OPENROUTER_BASE_URL` | Base URL for the API | `https://openrouter.ai/api/v1` |
| `LLM_RETRIES` | LLM call retries with exponential backoff | `3` |
| `VIDEO_LENGTH` | Default target video length in minutes | `1.0` |
| `EXPLANATION_DEPTH` | `basic`, `detailed`, or `comprehensive` | `detailed` |
| `VIDEO_ORIENTATION` | `landscape` or `portrait` | `landscape` |
| `DURATION_MODE` | `guide` (soft target) or `strict` (ffmpeg speed adjust) | `guide` |
| `MAX_RETRIES` | Maximum render error-correction attempts | `3` |
| `RENDER_TIMEOUT` | Render timeout in seconds | `120` |
| `RENDER_QUALITY` | `low` (480p), `medium` (720p), `high` (1080p), or manim flags `l`/`m`/`h`/`p`/`k` | `medium` |
| `RENDER_FPS` | Frame rate override | manim default |
| `STORYBOARD_ENABLED` | Plan timed sections before generating code | `true` |
| `VISUAL_QA_ENABLED` | Multimodal review of rendered frames with layout auto-fix | `false` |
| `VISUAL_QA_MAX_ATTEMPTS` | Layout-fix re-render attempts per job | `1` |
| `VISUAL_QA_FRAMES` | Frames sampled per visual QA review | `6` |
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `./chroma_db` |
| `CORS_ALLOW_ORIGINS` | Comma-separated allowed origins for the API | localhost dev origins |

## Usage

Everything goes through `start.sh`; no manual PATH exports are needed.

### Index the dataset

The Docker setup ships pre-seeded (see below). For a manual setup, populate the knowledge base once:

```bash
./start.sh index "./extracted data"
```

### Generate a video from the command line

```bash
./start.sh generate "Explain the concept of binary search"
```

| Flag | Description | Default |
|---|---|---|
| `-l`, `--length` | Target video length in minutes | from `.env` or `1.0` |
| `-d`, `--depth` | `basic`, `detailed`, `comprehensive` | from `.env` or `detailed` |
| `-t`, `--title` | Scene title | `Generated Scene` |
| `-o`, `--output` | Output directory for code and video | none |
| `-q`, `--quality` | Render quality: `low`, `medium`, `high` | from `.env` or `medium` |
| `--fps` | Frame rate override | manim default |
| `--orientation` | `landscape` or `portrait` | from `.env` or `landscape` |
| `--duration-mode` | `guide` or `strict` | from `.env` or `guide` |
| `--visual-qa` | Review rendered frames and auto-fix layout problems | off |

Example:

```bash
./start.sh generate "Red-Black Tree insertion" -l 3.0 -d comprehensive -q high --visual-qa -o ./output
```

Final videos are written to `output/` as `output_<id>.mp4`.

### Start the API server and web UI

```bash
./start.sh serve
```

The server binds to `http://0.0.0.0:8000` and serves the web interface at the root URL. The UI exposes prompt, length, depth, orientation, duration mode, render quality, live web research, and the visual QA pass, and shows per-job status, warnings, generated code, and a download link.

### Run via Docker

```bash
./start.sh --docker
```

This starts the application and a Neo4j instance via Docker Compose. The API is available at `http://localhost:8000` and the Neo4j browser at `http://localhost:7474`.

On first start, a one-shot `neo4j-seed` service loads the pre-indexed knowledge base from `neo4j-dumps/neo4j.dump` (about 1200 Manim examples with concept, class, and animation graph links), so no manual index run is needed. Seeding is skipped when the data volume already contains data; delete the `neo4j_data` volume to re-seed.

## REST API

### POST /generate

Starts a generation job and returns immediately with a job ID.

```json
{
  "topic": "How quicksort works",
  "length": 2.0,
  "depth": "detailed",
  "quality": "medium",
  "visual_qa": true,
  "title": "Quicksort Explained"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `topic` | string | yes (or `prompt`) | Subject of the video |
| `prompt` | string | yes (or `topic`) | Alias for `topic`; provide either one |
| `length` | float | no | Target length in minutes (0.1 to 30.0, default from `.env`) |
| `depth` | string | no | `basic`, `detailed`, or `comprehensive` |
| `title` | string | no | Scene title |
| `orientation` | string | no | `landscape` or `portrait` |
| `duration_mode` | string | no | `guide` or `strict` |
| `quality` | string | no | `low`, `medium`, or `high` |
| `fps` | int | no | Frame rate override (5 to 60) |
| `web_search` | bool | no | Ground the animation in live web research |
| `visual_qa` | bool | no | Review rendered frames and auto-fix layout |

### GET /jobs/{job_id}

Polls job status. `status` is one of `pending`, `running`, `completed`, `failed`. Non-fatal issues (for example a duration overshoot or a skipped review) are reported in `warnings`.

```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "video_path": "output/output_a1b2c3d4.mp4",
  "code": "from manim import *\n...",
  "error": null,
  "warnings": ["Duration mismatch: video is 25% longer than target (75.0s vs 60.0s)"]
}
```

### Other endpoints

| Method and path | Description |
|---|---|
| `GET /jobs/{job_id}/download` | Download the finished video (`video/mp4`) |
| `GET /search?query=...&limit=5` | Search the indexed Manim examples |
| `GET /health` | Health check |

## Project Structure

```
Logomotion/
  src/
    main.py              CLI entry point (index / generate / serve)
    api.py               FastAPI server and web UI host
    config.py            Centralized settings loaded from .env
    agent/
      graph.py           LangGraph state machine builder
      nodes.py           Workflow nodes (planning, generation, QA, merge)
      state.py           State type definitions
    graph_rag/
      indexer.py         JSONL dataset indexer (Neo4j + ChromaDB)
      retriever.py       Hybrid search (vector + graph traversal)
      embeddings.py      OpenRouter embedding function
      schema.py          Graph schema definitions
    manim_runner/
      executor.py        Manim subprocess execution
      validator.py       Static code validation and video integrity checks
      frames.py          Frame extraction for visual QA
    search/
      web.py             Live web research (DuckDuckGo + Wikipedia)
    tts/                 Kokoro TTS wrapper
  frontend/              Vue 3 web interface (built assets in src/static/)
  tests/                 Unit tests (no live databases or API keys required)
  neo4j-dumps/           Pre-indexed knowledge base dump for Docker seeding
  extracted data/        JSONL dataset for indexing
  docker-compose.yml     Application, Neo4j, and seed services
  start.sh               Single entry point script
  install.sh             Installation script
```

## Development

Run the test suite (no databases, network access, or API keys required; ffmpeg-dependent tests skip automatically when ffmpeg is absent):

```bash
venv/bin/python -m pytest tests/
```

The frontend is a Vue 3 + Vite app in `frontend/`. During development, `npm run dev` proxies API calls to a locally running server; production builds are committed to `src/static/` and served by FastAPI:

```bash
cd frontend && npm run build && cp -r dist/* ../src/static/
```

## License

See [LICENSE](LICENSE).
