FROM mcr.microsoft.com/devcontainers/python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    graphviz \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python development tools globally
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    pytest \
    pytest-cov \
    ruff \
    black \
    isort \
    mypy \
    pre-commit \
    sphinx \
    sphinx-rtd-theme \
    jupyter \
    notebook

# Set up git (will be overridden by postStartCommand)
RUN git config --global init.defaultBranch main

# Create workspace directory
WORKDIR /workspaces/scikit-plot

# Set the default user
USER vscode

# Configure bash history
RUN mkdir -p /home/vscode && \
    touch /home/vscode/.bash_history && \
    echo 'export HISTFILE=/home/vscode/.bash_history' >> /home/vscode/.bashrc

# Default command
CMD ["sleep", "infinity"]
