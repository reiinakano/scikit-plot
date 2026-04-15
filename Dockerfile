ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

COPY pyproject.toml ./
COPY scikitplot/ ./scikitplot/
COPY README.md LICENSE ./

RUN pip install --upgrade pip build wheel && \
    python -m build --wheel

FROM base AS production

COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser

CMD ["python", "-c", "import scikitplot; print('scikit-plot ready!')"]

FROM base AS development

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install -e ".[all]"

RUN pip install pre-commit && \
    pre-commit install || true

ENV MPLBACKEND=Agg

CMD ["/bin/bash"]

FROM base AS test

COPY pyproject.toml ./
COPY scikitplot/ ./scikitplot/
COPY README.md LICENSE ./

RUN pip install -e ".[test]"

ENV MPLBACKEND=Agg

CMD ["pytest", "-v", "--cov=scikitplot"]

FROM base AS docs

COPY pyproject.toml ./
COPY scikitplot/ ./scikitplot/
COPY docs/ ./docs/
COPY README.md LICENSE ./

RUN pip install -e ".[docs]"

WORKDIR /app/docs

CMD ["make", "html"]
