FROM python:3.12-slim

# Install OpenCV & video codec system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Ensure container virtual environment is used
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copy dependency manifest first for Docker layer caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Copy application source code and files
COPY . .

# Default runtime environment (override via .env or compose environment)
ENV EVAL_MODE=true \
    DEVICE=cpu \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

# Run FastAPI using uvicorn with hot reloading enabled for live container development
CMD ["sh", "-c", "uv run uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --reload"]
