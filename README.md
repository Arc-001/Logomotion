# Manim Graph RAG Agent

A specialized agentic system for generating mathematical and technical animations using Manim. This project integrates Graph RAG (Retrieval-Augmented Generation) with an autonomous LangGraph workflow to produce high-quality, compilable Manim code from natural language descriptions.

## Overview

The system operates by retrieving relevant Manim code patterns from a knowledge base (indexed from documentation and examples) and using a Large Language Model to synthesize new scenes. It features a self-correcting execution loop that handles compilation errors and runtime exceptions automatically.

### Architecture

![System Architecture](assets/architecture.png)

1.  **Knowledge Base**: Hybrid index using Neo4j (Graph) and ChromaDB (Vector) for storing Manim documentation and examples.
2.  **Agent Workflow**: LangGraph-based state machine that manages planning, code generation, execution, and error recovery.
3.  **Rendering Engine**: Manim Community Edition running in a containerized environment.
4.  **Audio Synthesis**: Integrated Kokoro TTS for generating voiceovers synchronized with the animation.

## Prerequisites

*   Docker and Docker Compose
*   Python 3.10+ (for local execution without Docker)
*   OpenAI/OpenRouter API key

## Configuration

Environment variables are managed via the `.env` file. Key configurations include:

*   `OPENROUTER_API_KEY`: API key for the LLM provider.
*   `VIDEO_LENGTH`: Default target video duration in minutes.
*   `EXPLANATION_DEPTH`: Default complexity level (basic, detailed, comprehensive).
*   `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Database connection details.

## Usage

### 1. Initialization

Start the infrastructure services (Neo4j, etc.):

```bash
docker-compose up -d
```

### 2. Indexing Data

Before generating videos, index the reference dataset to populate the Graph RAG system:

```bash
./start.sh index --data "./path/to/extracted_data"
```

### 3. Generating Animations

Generate a video from a text prompt.

```bash
./start.sh generate "Explain the concept of binary search"
```

**Options:**

*   `-l`, `--length`: Target video length in minutes (e.g., `-l 2.0`).
*   `-d`, `--depth`: Explanation depth (`basic`, `detailed`, `comprehensive`).
*   `-o`, `--output`: output directory.

Example:

```bash
./start.sh generate "Red-Black Tree Insertion" -l 3.0 -d comprehensive
```

### 4. API Server

Start the REST API for remote interaction:

```bash
./start.sh serve
```

## Detailed Workflow

1.  **Query Analysis**: The user's prompt is analyzed to identify key concepts.
2.  **Context Retrieval**:
    *   **Vector Search (ChromaDB)**: Finds semantically similar examples.
    *   **Graph Traversal (Neo4j)**: Retrieves related class hierarchies and dependencies.
3.  **Code Synthesis**: The LLM generates a complete Python script using the retrieved context.
4.  **Iterative Execution**:
    *   The system attempts to render the code.
    *   If rendering fails, the error output is fed back to the LLM for correction (Recorrector Node).
    *   This loop repeats until success or maximum retries are reached.
5.  **Post-Processing**:
    *   Audio scripts are generated and synthesized using Kokoro TTS.
    *   Audio is merged with the video output.

## Demos

<video src="assets/1.mp4" width="600" controls></video>

<video src="assets/2.mp4" width="600" controls></video>

<video src="assets/3.mp4" width="600" controls></video>
