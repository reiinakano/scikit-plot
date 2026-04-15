.PHONY: help install install-dev install-all test test-cov lint format clean build docs docs-serve docker-build docker-run docker-test ci-setup ci-test ci-lint ci-format release changelog

# Default target
help:
	@echo "Available targets:"
	@echo "  install        - Install package with basic dependencies"
	@echo "  install-dev    - Install package with development dependencies"
	@echo "  install-all    - Install package with all dependencies"
	@echo "  test           - Run tests"
	@echo "  test-cov       - Run tests with coverage"
	@echo "  test-parallel  - Run tests in parallel"
	@echo "  lint           - Run linting checks (ruff, mypy)"
	@echo "  format         - Format code (black, isort)"
	@echo "  format-check   - Check code formatting without modifying"
	@echo "  clean          - Clean build artifacts"
	@echo "  build          - Build package distributions"
	@echo "  docs           - Build documentation"
	@echo "  docs-serve     - Serve documentation locally"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-test    - Run tests in Docker"
	@echo "  pre-commit     - Install and run pre-commit hooks"
	@echo "  ci-setup       - Setup for CI environment"
	@echo "  ci-test        - Run CI tests"
	@echo "  ci-lint        - Run CI linting"
	@echo "  release        - Prepare for release"
	@echo "  changelog      - Generate changelog"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-all:
	pip install -e ".[all]"
	pre-commit install

# Testing targets
test:
	pytest scikitplot/tests -v

test-cov:
	pytest scikitplot/tests -v --cov=scikitplot --cov-report=term-missing --cov-report=html --cov-report=xml

test-parallel:
	pytest scikitplot/tests -v -n auto

test-ci:
	pytest scikitplot/tests -v --cov=scikitplot --cov-report=xml --cov-fail-under=80

# Linting and formatting targets
lint:
	ruff check scikitplot
	mypy scikitplot --ignore-missing-imports

lint-fix:
	ruff check scikitplot --fix

format:
	black scikitplot examples
	isort scikitplot examples

format-check:
	black --check scikitplot examples
	isort --check-only scikitplot examples

# Cleaning targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	rm -rf .venv/
	rm -rf .tox/

# Build targets
build:
	python -m build

build-check:
	twine check dist/*

# Documentation targets
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

docs-serve: docs
	cd docs/_build/html && python -m http.server 8000

docs-live:
	cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

# Docker targets
docker-build:
	docker build -t scikit-plot:latest -t scikit-plot:$(shell python -c "import scikitplot; print(scikitplot.__version__)") .

docker-run:
	docker run -it --rm scikit-plot:latest

docker-test:
	docker run --rm scikit-plot:latest make test

docker-dev:
	docker-compose up -d dev

docker-dev-down:
	docker-compose down

# Pre-commit targets
pre-commit:
	pre-commit install
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# CI targets
ci-setup:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev,test]"

ci-test:
	pytest scikitplot/tests -v --cov=scikitplot --cov-report=xml --cov-report=term

ci-lint:
	ruff check scikitplot
	black --check scikitplot
	isort --check-only scikitplot

ci-type-check:
	mypy scikitplot --ignore-missing-imports

# Release targets
release-check:
	python -m build
	twine check dist/*

release-test: release-check
	twine upload --repository testpypi dist/*

release: release-check
	twine upload dist/*

# Version bumping (requires bump2version)
bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major

# Development server
jupyter:
	jupyter notebook examples/jupyter_notebooks/

# Environment setup
venv:
	python -m venv .venv
	@echo "Run 'source .venv/bin/activate' (Linux/Mac) or '.venv\\Scripts\\activate' (Windows)"

conda-env:
	conda env create -f environment.yml
	@echo "Run 'conda activate scikit-plot-env'"
