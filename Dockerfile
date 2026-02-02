# ============================================================================
# Manim Graph RAG Agent - Dockerfile
# Multi-stage build for optimized image size
# ============================================================================

# Stage 1: Base with system dependencies
FROM python:3.11-slim as base

# Install system dependencies for Manim
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    pkg-config \
    # Manim dependencies
    libcairo2-dev \
    libpango1.0-dev \
    libglib2.0-dev \
    libffi-dev \
    # Media processing
    ffmpeg \
    # LaTeX (minimal for math rendering)
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-fonts-extra \
    dvipng \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Stage 2: Python dependencies
FROM base as deps

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final image
FROM deps as final

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY README.md .

# Create directories
RUN mkdir -p /app/output /app/chroma_db

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CHROMA_PERSIST_DIRECTORY=/app/chroma_db
ENV MANIM_OUTPUT_DIR=/app/output

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.main"]
