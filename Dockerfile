# =============================================================================
# Multi-stage Dockerfile for scikit-plot
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with common dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim as base

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
COPY pyproject.toml ./
RUN pip install -e ".[dev]"

# Copy source code
COPY . .

# Install pre-commit hooks
RUN pre-commit install

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["bash"]

# -----------------------------------------------------------------------------
# Stage 3: Testing environment
# -----------------------------------------------------------------------------
FROM base as testing

# Install test dependencies
COPY pyproject.toml ./
RUN pip install -e ".[test]"

# Copy source code
COPY . .

# Run tests by default
CMD ["pytest", "-v", "--cov=scikitplot"]

# -----------------------------------------------------------------------------
# Stage 4: Production build
# -----------------------------------------------------------------------------
FROM base as builder

# Install build dependencies
RUN pip install --upgrade pip build hatchling hatch-vcs

# Copy source code
COPY . .

# Build package
RUN python -m build

# -----------------------------------------------------------------------------
# Stage 5: Production runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy built package from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Create non-root user for security
RUN useradd -m -u 1000 scikitplot && chown -R scikitplot:scikitplot /app
USER scikitplot

# Default command
CMD ["python", "-c", "import scikitplot; print(f'scikit-plot {scikitplot.__version__}')"]

# -----------------------------------------------------------------------------
# Stage 6: Documentation build
# -----------------------------------------------------------------------------
FROM base as docs

# Install documentation dependencies
COPY pyproject.toml ./
RUN pip install -e ".[docs]"

# Copy source code
COPY . .

# Build documentation
WORKDIR /app/docs
RUN make html

# Serve documentation (optional)
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--directory", "_build/html"]
