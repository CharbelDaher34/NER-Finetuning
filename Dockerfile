# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv in a single layer
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install Python dependencies using uv
RUN uv sync --frozen
RUN uv run python -m ensurepip


# Copy application files and model directory
COPY convert_to_gguf.py api.py dataset.jsonl ./
COPY best_model/ ./best_model/

# Run conversion script if model doesn't exist, then verify
RUN if [ ! -f best_model/model.gguf ]; then \
        echo "Converting model to GGUF format..." && \
        uv run convert_to_gguf.py; \
    fi && \
    test -f best_model/model.gguf || (echo "ERROR: Model file not found!" && exit 1) && \
    echo "✓ Model verified: $(du -h best_model/model.gguf | cut -f1)"

# Expose API port
EXPOSE 8347

# Health check - increased start period for large model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8347/health || exit 1

# Run the API
CMD ["uv", "run", "api.py"]
