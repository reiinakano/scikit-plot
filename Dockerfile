# =============================================================================
# Multi-stage Dockerfile for scikit-plot
# Supports: development, testing, and production environments
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with common dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Development environment
# -----------------------------------------------------------------------------
FROM base as development

# Install development dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install -e ".[dev,docs,test]" && \
    pip install jupyter ipykernel

# Copy source code
COPY . .

# Install pre-commit hooks
RUN pre-commit install || true

# Expose port for Jupyter
EXPOSE 8888

# Default command for development
CMD ["bash"]

# -----------------------------------------------------------------------------
# Stage 3: Testing environment
# -----------------------------------------------------------------------------
FROM base as testing

# Install test dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install -e ".[test]"

# Copy source code
COPY . .

# Run tests
CMD ["pytest", "scikitplot/tests", "-v", "--cov=scikitplot", "--cov-report=term-missing"]

# -----------------------------------------------------------------------------
# Stage 4: Production/Release environment
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install only runtime dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install -e "."

# Copy only necessary source files
COPY scikitplot/ ./scikitplot/
COPY README.md LICENSE ./

# Create non-root user for security
RUN useradd -m -u 1000 scikitplot && \
    chown -R scikitplot:scikitplot /app
USER scikitplot

# Default command
CMD ["python", "-c", "import scikitplot; print(f'scikit-plot v{scikitplot.__version__}')"]

# -----------------------------------------------------------------------------
# Stage 5: Documentation build environment
# -----------------------------------------------------------------------------
FROM base as docs

# Install documentation dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install -e ".[docs]"

# Copy source code
COPY . .

# Build documentation
RUN cd docs && make html

# Serve documentation
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--directory", "docs/_build/html"]
