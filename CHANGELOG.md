# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modernized build system with `pyproject.toml` (PEP 517/518)
- Comprehensive CI/CD pipeline with GitHub Actions
- Docker and Docker Compose support for development and production
- Pre-commit hooks for code quality
- VS Code development container configuration
- Makefile for common development tasks
- Updated testing infrastructure with pytest and coverage
- Type checking with mypy
- Linting with ruff
- Code formatting with black and isort

### Changed
- Migrated from `setup.py` to `pyproject.toml`
- Updated minimum Python version to 3.9
- Updated dependencies to modern versions
- Improved documentation build process

### Deprecated
- Python 2.7 support removed
- Python 3.5, 3.6, 3.7, 3.8 support removed

### Removed
- Travis CI configuration (replaced by GitHub Actions)
- Legacy `setup.py` based installation

### Security
- Added security scanning with Bandit
- Added container vulnerability scanning with Trivy

## [0.3.7] - 2018-XX-XX

### Added
- Initial release with core plotting functionality
- Support for various machine learning visualizations
- Integration with scikit-learn

[Unreleased]: https://github.com/reiinakano/scikit-plot/compare/v0.3.7...HEAD
[0.3.7]: https://github.com/reiinakano/scikit-plot/releases/tag/v0.3.7
