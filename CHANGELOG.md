# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modernized build system with `pyproject.toml` (PEP 517/518)
- Added comprehensive CI/CD pipeline with GitHub Actions
- Added pre-commit hooks for code quality
- Added Docker and Docker Compose support
- Added VS Code development container configuration
- Added automated testing with pytest and coverage reporting
- Added linting with ruff, black, and isort
- Added type checking with mypy
- Added security scanning with Bandit and CodeQL

### Changed
- Migrated from `setup.py` to `pyproject.toml`
- Updated minimum Python version to 3.9
- Updated dependencies to modern versions
- Improved documentation configuration

### Deprecated
- Python 2.7 support removed
- Python 3.5-3.8 support removed

### Removed
- Legacy `.travis.yml` configuration
- Legacy `tox.ini` configuration
- Legacy `setup.py` (functionality moved to `pyproject.toml`)

## [0.3.7] - 2018-XX-XX

### Added
- Initial release
- Basic plotting functionality for scikit-learn
- Support for confusion matrix, ROC curves, precision-recall curves
- Support for learning curves, validation curves
- Support for clustering visualization
- Support for decomposition visualization

[Unreleased]: https://github.com/reiinakano/scikit-plot/compare/v0.3.7...HEAD
[0.3.7]: https://github.com/reiinakano/scikit-plot/releases/tag/v0.3.7
