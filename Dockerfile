# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install Python dependencies using uv
RUN uv sync --frozen

# Copy application files
COPY convert_to_gguf.py api.py test_inference.py ./
COPY dataset.jsonl test_dataset.jsonl ./

# Copy the best_model directory (includes model.gguf)
COPY best_model/ ./best_model/

# Copy llama.cpp if needed (for conversion script)
# Note: If model.gguf already exists, conversion can be skipped
COPY llama.cpp/ ./llama.cpp/

# Run conversion script (will skip if model already exists)
RUN uv run convert_to_gguf.py || echo "Conversion skipped or failed - using existing model"

# Expose API port
EXPOSE 8347

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8347/health || exit 1

# Run the API
CMD ["uv", "run", "api.py"]

