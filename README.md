# Logomotion

An agentic system that generates narrated mathematical and technical animations from natural language prompts. It combines Graph RAG retrieval with a self-correcting LangGraph workflow to produce compilable Manim code, render it, synthesize voiceover audio, and merge everything into a final video.

## Architecture

![System Architecture](assets/architecture.png)

The pipeline is built as a LangGraph state machine with seven nodes:

```
START
  |
  v
[video_code_gen] ---- retrieves examples from Graph RAG (Neo4j + ChromaDB)
  |                                          |
  v                                          v
[code_executor]                   [transcript_processor]
  |                                          |
  +-- error? --> [recorrector] --+            |   (Kokoro TTS)
  |              (loops until    |            |
  |               success or     |            |
  |               max retries)   |            |
  v                              |            |
[render_checker]                              |
  |                                           |
  v                                           |
[synchronizer] <------------------------------+
  |
  v
[audio_video_merger]  (ffmpeg)
  |
  v
 END --> final .mp4 with narration
```

1. **video_code_gen** -- Retrieves similar Manim examples via hybrid search (vector + graph), builds a prompt with positioning/API constraints, and calls the LLM to produce scene code and a timestamped transcript.
2. **code_executor** -- Writes the code to a temp file and runs `manim render`. On failure, passes the error downstream.
3. **recorrector** -- Receives the error output and the original code, asks the LLM to fix it, and sends the corrected code back to the executor. Loops up to `MAX_RETRIES` times (default 5).
4. **transcript_processor** -- Runs in parallel with the code branch. Splits the transcript into segments and synthesizes each one with Kokoro TTS (82M parameter model, runs locally on CPU).
5. **render_checker** -- Validates the rendered video (file size, ffprobe).
6. **synchronizer** -- Joins the video and audio branches.
7. **audio_video_merger** -- Concatenates all audio segments and merges them with the video via ffmpeg.

## Prerequisites

- Python 3.10+
- System packages for Manim: `libcairo2-dev`, `libpango1.0-dev`, `ffmpeg`, and a LaTeX distribution (see `Dockerfile` for the full list)
- An OpenRouter API key (or any OpenAI-compatible endpoint)
- Neo4j 5+ (can run via Docker)

## Installation

```bash
git clone https://github.com/Arc-001/Logomotion.git
cd Logomotion
./install.sh
```

This creates a virtual environment, installs all Python dependencies (including Manim), and sets up output directories. Edit `.env` before running.

For Docker-based installation:

```bash
./install.sh --docker
```

## Configuration

All configuration lives in `.env`. Key variables:

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | API key for the LLM provider | (required) |
| `OPENROUTER_MODEL` | Model identifier | `google/gemini-3-flash-preview` |
| `OPENROUTER_BASE_URL` | Base URL for the API | `https://openrouter.ai/api/v1` |
| `VIDEO_LENGTH` | Default target video length in minutes | `1.0` |
| `EXPLANATION_DEPTH` | Default depth: `basic`, `detailed`, or `comprehensive` | `detailed` |
| `MAX_RETRIES` | Maximum error-correction attempts | `5` |
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `./chroma_db` |
| `MANIM_OUTPUT_DIR` | Output directory for rendered files | `./output` |

## Usage

Everything goes through `start.sh`. No manual PATH exports are needed.

### Index the dataset

Before generating videos, populate the Graph RAG knowledge base:

```bash
./start.sh index "./extracted data"
```

### Generate a video (CLI)

```bash
./start.sh generate "Explain the concept of binary search"
```

Options:

| Flag | Description | Default |
|---|---|---|
| `-l`, `--length` | Target video length in minutes | from `.env` or `1.0` |
| `-d`, `--depth` | Explanation depth: `basic`, `detailed`, `comprehensive` | from `.env` or `detailed` |
| `-o`, `--output` | Output directory for code and video | none |
| `-t`, `--title` | Scene title | `Generated Scene` |

Example with all options:

```bash
./start.sh generate "Red-Black Tree insertion" -l 3.0 -d comprehensive -o ./output
```

Output files are written to the `output/` directory: `scene.py` (the generated Manim code) and `output.mp4` (the final video with narration).

### Start the API server

```bash
./start.sh serve
```

The server binds to `http://0.0.0.0:8000` by default.

### Run via Docker

```bash
./start.sh --docker
```

This starts the application container and a Neo4j instance via Docker Compose. The API is available at `http://localhost:8000` and the Neo4j browser at `http://localhost:7474`.

## REST API

### POST /generate

Start a video generation job. Returns immediately with a job ID.

**Request body:**

```json
{
  "topic": "How quicksort works",
  "length": 2.0,
  "depth": "detailed",
  "title": "Quicksort Explained"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `topic` | string | yes (or `prompt`) | Subject of the video |
| `prompt` | string | yes (or `topic`) | Alias for `topic` -- provide either one |
| `length` | float | no | Target length in minutes (0.1--30.0, default from `.env`) |
| `depth` | string | no | `basic`, `detailed`, or `comprehensive` (default from `.env`) |
| `title` | string | no | Scene title (default: `Generated Scene`) |

**Response:**

```json
{
  "job_id": "a1b2c3d4",
  "status": "pending",
  "message": "Job queued for processing"
}
```

### GET /jobs/{job_id}

Poll for job status.

```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "video_path": "/tmp/manim_merge_.../final_with_audio.mp4",
  "code": "from manim import *\n...",
  "error": null
}
```

Status values: `pending`, `running`, `completed`, `failed`.

### GET /jobs/{job_id}/download

Download the generated video file directly. Returns `video/mp4`. Only available when the job status is `completed`.

### GET /search?query=...&limit=5

Search the indexed Manim examples. Returns matching examples with scores, used classes, and used animations.

### GET /health

Health check. Returns `{"status": "ok"}`.

## Project Structure

```
Logomotion/
  src/
    main.py              -- CLI entry point (index / generate / serve)
    api.py               -- FastAPI server
    agent/
      graph.py           -- LangGraph state machine builder
      nodes.py           -- All seven workflow nodes
      state.py           -- State type definitions
    graph_rag/
      indexer.py         -- JSONL dataset indexer (Neo4j + ChromaDB)
      retriever.py       -- Hybrid search (vector + graph traversal)
      embeddings.py      -- Embedding configuration
      schema.py          -- Graph schema definitions
    manim_runner/
      executor.py        -- Manim subprocess execution with auto-discovery
      validator.py       -- Code validation
    tts/
      __init__.py        -- Kokoro TTS wrapper
  start.sh               -- Single entry point script
  install.sh             -- Installation script
  Dockerfile             -- Multi-stage Docker build
  docker-compose.yml     -- Application + Neo4j services
  requirements.txt       -- Python dependencies
  extracted data/        -- JSONL dataset for indexing
  output/                -- Generated videos and code
  chroma_db/             -- ChromaDB persistent storage
```

## Demos

<video src="https://github.com/Arc-001/Logomotion/blob/3bd3ca4ee2fdc92c05d77e7390107ad60589ed0f/assets/1.mp4" width="600" controls></video>

<video src="https://github.com/Arc-001/Logomotion/blob/3bd3ca4ee2fdc92c05d77e7390107ad60589ed0f/assets/2.mp4" width="600" controls></video>

<video src="https://github.com/Arc-001/Logomotion/blob/3bd3ca4ee2fdc92c05d77e7390107ad60589ed0f/assets/3.mp4" width="600" controls></video>

## License

See [LICENSE](LICENSE).
