.PHONY: help install install-dev install-test install-docs install-all
.PHONY: clean clean-build clean-pyc clean-test clean-docs
.PHONY: lint format check test test-cov test-all
.PHONY: build build-docs docs serve-docs
.PHONY: docker-build docker-test docker-dev docker-jupyter
.PHONY: pre-commit setup release

help:
	@echo "scikit-plot - Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Installation:"
	@echo "  install          Install the package"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-test     Install with test dependencies"
	@echo "  install-docs     Install with documentation dependencies"
	@echo "  install-all      Install all dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run all linters (ruff, mypy)"
	@echo "  format           Format code with black and isort"
	@echo "  check            Run format check without modifying files"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run tests"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  test-all         Run tests on all Python versions (requires tox)"
	@echo ""
	@echo "Building:"
	@echo "  build            Build the package"
	@echo "  build-docs       Build documentation"
	@echo "  docs             Build and serve documentation locally"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-test      Run tests in Docker"
	@echo "  docker-dev       Start development environment"
	@echo "  docker-jupyter   Start Jupyter Lab in Docker"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean all generated files"
	@echo "  pre-commit       Install and run pre-commit hooks"
	@echo "  setup            Initial project setup"
	@echo "  release          Create a new release"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-test:
	pip install -e ".[test]"

install-docs:
	pip install -e ".[docs]"

install-all:
	pip install -e ".[all]"

clean: clean-build clean-pyc clean-test clean-docs

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf *.egg-info/
	rm -rf *.egg
	rm -rf .eggs

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*~' -exec rm -f {} +

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .hypothesis/

clean-docs:
	rm -rf docs/_build/
	rm -rf site/

lint:
	ruff check .
	mypy scikitplot --ignore-missing-imports

format:
	black .
	isort .

check:
	black --check .
	isort --check-only --diff .
	ruff check .

test:
	pytest -v

test-cov:
	pytest -v --cov=scikitplot --cov-report=term-missing --cov-report=html --cov-report=xml

test-all:
	tox

build:
	python -m build
	twine check dist/*

build-docs:
	cd docs && make clean html

docs: build-docs
	cd docs/_build/html && python -m http.server 8000

serve-docs:
	cd docs/_build/html && python -m http.server 8000

docker-build:
	docker-compose build

docker-test:
	docker-compose run --rm test

docker-dev:
	docker-compose run --rm development

docker-jupyter:
	docker-compose up jupyter

pre-commit:
	pre-commit install
	pre-commit run --all-files

setup: install-all pre-commit
	@echo "Project setup complete!"

release:
	@echo "Creating release..."
	@read -p "Enter version (e.g., 0.3.8): " version; \
	git tag -a v$$version -m "Release v$$version"; \
	git push origin v$$version

.PHONY: docker-clean
docker-clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

.PHONY: docker-logs
docker-logs:
	docker-compose logs -f

.PHONY: update-deps
update-deps:
	pip install --upgrade pip
	pip-compile --upgrade pyproject.toml

.PHONY: security
security:
	pip install safety
	safety check

.PHONY: benchmark
benchmark:
	pytest -v --benchmark-only

.DEFAULT_GOAL := help
