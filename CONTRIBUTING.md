[![PyPI version](https://badge.fury.io/py/scikit-plot.svg)](https://badge.fury.io/py/scikit-plot)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()
[![Build Status](https://github.com/reiinakano/scikit-plot/workflows/CI/badge.svg)](https://github.com/reiinakano/scikit-plot/actions)
[![PyPI](https://img.shields.io/pypi/pyversions/scikit-plot.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.293191.svg)](https://doi.org/10.5281/zenodo.293191)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/reiinakano/scikit-plot/branch/main/graph/badge.svg)](https://codecov.io/gh/reiinakano/scikit-plot)

# Contributing to scikit-plot

Thank you for your interest in contributing to scikit-plot! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Quick Start

1. **Fork and Clone**

   ```bash
   git clone https://github.com/YOUR_USERNAME/scikit-plot.git
   cd scikit-plot
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Development Dependencies**

   ```bash
   pip install -e ".[all]"
   pre-commit install
   ```

4. **Verify Installation**

   ```bash
   python -c "import scikitplot; print('Installation successful!')"
   pytest
   ```

### Using Docker

Alternatively, you can use Docker for development:

```bash
# Build and start development container
docker-compose up development

# Run tests in container
docker-compose run test

# Start Jupyter Lab
docker-compose up jupyter
```

### Using VS Code Dev Containers

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the project in VS Code
3. Press `F1` and select "Dev Containers: Reopen in Container"

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
# or
git checkout -b docs/your-documentation-update
```

### 2. Make Changes

- Write clean, readable code
- Follow the [Code Standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Checks

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test-cov

# Run all checks
make check
```

### 4. Commit Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new plot type for feature importance"
git commit -m "fix: correct ROC curve calculation for multiclass"
git commit -m "docs: update installation instructions"
git commit -m "test: add tests for plot_confusion_matrix"
git commit -m "refactor: simplify plot_roc implementation"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 100)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [Ruff](https://docs.astral.sh/ruff/) for linting

### Code Quality Tools

```bash
# Format code
black .
isort .

# Check code
ruff check .

# Type check (optional)
mypy scikitplot
```

### Import Order

```python
# Standard library
import os
import sys

# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

# Local imports
from scikitplot.helpers import validate_labels
```

### Docstrings

Use NumPy-style docstrings:

```python
def plot_roc(y_true, y_probas, title='ROC Curve', ax=None, 
             figsize=None, title_fontsize='large', text_fontsize='medium'):
    """
    Plot ROC curves for binary or multiclass classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_probas : array-like
        Probability estimates of the positive class.
    title : str, optional
        Title of the plot. Default is 'ROC Curve'.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto.
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> import scikitplot as skplt
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.naive_bayes import GaussianNB
    >>> X, y = load_digits(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> nb = GaussianNB()
    >>> nb.fit(X_train, y_train)
    >>> skplt.metrics.plot_roc(y_test, nb.predict_proba(X_test))
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scikitplot --cov-report=html

# Run specific test file
pytest scikitplot/tests/test_metrics.py

# Run specific test
pytest scikitplot/tests/test_metrics.py::test_plot_roc

# Run tests in parallel
pytest -n auto
```

### Writing Tests

```python
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import scikitplot as skplt


class TestPlotROC:
    """Tests for plot_roc function."""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification test data."""
        X, y = make_classification(n_samples=100, n_features=20, 
                                   n_classes=2, random_state=42)
        return X, y
    
    def test_plot_roc_binary(self, binary_classification_data):
        """Test ROC curve for binary classification."""
        X, y = binary_classification_data
        clf = LogisticRegression()
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)
        
        ax = skplt.metrics.plot_roc(y, y_proba)
        
        assert ax is not None
        assert ax.get_xlabel() == 'False Positive Rate'
        assert ax.get_ylabel() == 'True Positive Rate'
```

### Test Guidelines

- Use descriptive test names
- Use fixtures for common test data
- Test edge cases and error conditions
- Ensure tests are independent
- Use parameterized tests for multiple scenarios

## Documentation

### Building Documentation

```bash
cd docs
make html
```

### Documentation Guidelines

- Keep docstrings up to date
- Include examples in docstrings
- Update user guide for new features
- Add API documentation for new functions/classes

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Run all linters and formatters
3. Update documentation
4. Add changelog entry (if applicable)
5. Request review from maintainers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Changelog updated (if needed)
- [ ] PR description is clear and complete

### Review Process

1. Automated checks must pass (CI)
2. At least one maintainer approval required
3. All conversations must be resolved
4. Branch must be up to date with main

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Creating a Release

1. Update version in `pyproject.toml` and `scikitplot/__init__.py`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag -a v0.3.8 -m "Release v0.3.8"`
4. Push tag: `git push origin v0.3.8`
5. GitHub Actions will automatically:
   - Build the package
   - Run tests
   - Publish to PyPI
   - Create GitHub Release

## Getting Help

- Open a [Discussion](https://github.com/reiinakano/scikit-plot/discussions) for questions
- Open an [Issue](https://github.com/reiinakano/scikit-plot/issues) for bugs
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to scikit-plot! 🎉
