.PHONY: help install install-dev install-test install-docs install-notebook test test-cov test-notebook test-notebook-fast test-notebook-full lint format clean build docs docs-serve docker-build docker-dev docker-test all

# Default target
help:
	@echo "Available targets:"
	@echo "  install           - Install the package"
	@echo "  install-dev       - Install with development dependencies"
	@echo "  install-test      - Install with test dependencies"
	@echo "  install-docs      - Install with documentation dependencies"
	@echo "  install-notebook  - Install with notebook dependencies"
	@echo "  test              - Run tests"
	@echo "  test-cov          - Run tests with coverage"
	@echo "  test-notebook     - Run notebook tests (fast)"
	@echo "  test-notebook-fast- Run quick notebook sanity checks"
	@echo "  test-notebook-full- Run all notebook tests with execution"
	@echo "  lint              - Run linting checks"
	@echo "  format            - Format code"
	@echo "  clean             - Clean build artifacts"
	@echo "  build             - Build package"
	@echo "  docs              - Build documentation"
	@echo "  docs-serve        - Serve documentation locally"
	@echo "  docker-build      - Build Docker images"
	@echo "  docker-dev        - Run development container"
	@echo "  docker-test       - Run tests in container"
	@echo "  all               - Run lint, test, and build"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-test:
	pip install -e ".[test]"

install-docs:
	pip install -e ".[docs]"

install-notebook:
	pip install -e ".[notebook]"

# Testing targets
test:
	pytest -v

test-cov:
	pytest -v --cov=scikitplot --cov-report=term-missing --cov-report=html

test-fast:
	pytest -v -m "not slow"

test-notebook:
	pytest scikitplot/tests/notebooks/test_notebooks.py -v

test-notebook-fast:
	pytest scikitplot/tests/notebooks/test_notebooks.py -v -m "not notebook"

test-notebook-full:
	pytest scikitplot/tests/notebooks/test_notebooks.py -v --tb=short

# Linting and formatting targets
lint:
	ruff check scikitplot
	mypy scikitplot --ignore-missing-imports

format:
	ruff check --fix scikitplot
	ruff format scikitplot
	isort scikitplot

format-check:
	ruff format --check scikitplot
	isort --check-only scikitplot

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Cleaning targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Build targets
build:
	python -m build

build-check:
	twine check dist/*
	check-wheel-contents dist/*.whl

# Documentation targets
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

docs-serve:
	cd docs/_build/html && python -m http.server 8000

docs-live:
	cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

# Docker targets
docker-build:
	docker-compose build

docker-dev:
	docker-compose up -d dev
	docker-compose logs -f dev

docker-test:
	docker-compose run --rm test

docker-prod:
	docker-compose up prod

docker-docs:
	docker-compose up -d docs

docker-down:
	docker-compose down -v

docker-clean:
	docker-compose down -v --rmi all

# Release targets
version:
	python -c "import scikitplot; print(scikitplot.__version__)"

release-check:
	python -m build
	twine check dist/*

release-test:
	twine upload --repository testpypi dist/*

release:
	twine upload dist/*

# CI targets (used in GitHub Actions)
ci: lint test build

# All-in-one target
all: clean install-dev lint test-cov build
	@echo "All tasks completed successfully!"

# Development workflow
dev-setup: clean install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make lint' to check code quality"
	@echo "Run 'make format' to format code"
