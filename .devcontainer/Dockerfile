FROM mcr.microsoft.com/devcontainers/python:3.11

# Install additional system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        vim \
        nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install development tools globally
RUN pip install \
    black \
    isort \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pre-commit \
    jupyter \
    ipykernel

# Set the default user
USER vscode

# Configure git
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

# Set working directory
WORKDIR /workspaces/scikit-plot
