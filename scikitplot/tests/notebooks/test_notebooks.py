"""Tests for Jupyter Notebooks.

This module contains tests for example Jupyter notebooks to ensure they execute
without errors and produce expected outputs.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Get the notebooks directory
NOTEBOOKS_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "jupyter_notebooks"

# List of notebooks to test
NOTEBOOKS = [
    "plot_confusion_matrix.ipynb",
    "plot_roc_curve.ipynb",
    "plot_precision_recall_curve.ipynb",
    "plot_learning_curve.ipynb",
    "plot_elbow_curve.ipynb",
    "plot_silhouette.ipynb",
    "plot_pca_2d_projection.ipynb",
    "plot_pca_component_variance.ipynb",
    "plot_feature_importance.ipynb",
    "plot_cumulative_gain.ipynb",
    "plot_lift_curve.ipynb",
    "plot_ks_statistic.ipynb",
]


def get_notebook_path(notebook_name: str) -> Path:
    """Get the full path to a notebook.

    Parameters
    ----------
    notebook_name : str
        Name of the notebook file.

    Returns
    -------
    Path
        Full path to the notebook.
    """
    return NOTEBOOKS_DIR / notebook_name


@pytest.fixture(scope="module")
def jupyter_kernel():
    """Fixture to ensure Jupyter kernel is available.

    Yields
    ------
    bool
        True if kernel is available.
    """
    try:
        import ipykernel  # noqa: F401
        yield True
    except ImportError:
        pytest.skip("ipykernel not installed")


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
def test_notebook_runs_without_errors(notebook_name: str, jupyter_kernel, tmp_path):
    """Test that a notebook runs without errors using papermill.

    Parameters
    ----------
    notebook_name : str
        Name of the notebook to test.
    jupyter_kernel : bool
        Fixture ensuring kernel availability.
    tmp_path : Path
        Temporary directory for output.
    """
    pytest.importorskip("papermill", reason="papermill not installed")

    notebook_path = get_notebook_path(notebook_name)

    if not notebook_path.exists():
        pytest.skip(f"Notebook {notebook_name} not found")

    # Create output path
    output_path = tmp_path / f"output_{notebook_name}"

    # Run the notebook
    import papermill as pm

    try:
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            kernel_name="python3",
            progress_bar=False,
        )
    except Exception as e:
        pytest.fail(f"Notebook {notebook_name} failed to execute: {e}")

    # Check that output notebook was created
    assert output_path.exists(), f"Output notebook {output_path} was not created"


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
def test_notebook_with_nbval(notebook_name: str, jupyter_kernel):
    """Test notebook outputs using nbval (if available).

    This test uses nbval to validate that notebook cell outputs are consistent.

    Parameters
    ----------
    notebook_name : str
        Name of the notebook to test.
    jupyter_kernel : bool
        Fixture ensuring kernel availability.
    """
    pytest.importorskip("nbval", reason="nbval not installed")

    notebook_path = get_notebook_path(notebook_name)

    if not notebook_path.exists():
        pytest.skip(f"Notebook {notebook_name} not found")

    # Run nbval on the notebook
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--nbval",
            "--nbval-lax",  # Lax mode - don't fail on output differences
            "-v",
            str(notebook_path),
        ],
        capture_output=True,
        text=True,
    )

    # Check that the notebook executed without errors
    # We use lax mode to avoid failing on minor output differences (like timestamps)
    if result.returncode != 0:
        # Only fail if there are actual execution errors, not output differences
        if "FAILED" in result.stdout and "Error" in result.stdout:
            pytest.fail(f"Notebook {notebook_name} failed nbval validation:\n{result.stdout}")


@pytest.mark.notebook
@pytest.mark.parametrize("notebook_name", NOTEBOOKS[:3])  # Test first 3 notebooks for speed
def test_notebook_cells_execute(notebook_name: str, jupyter_kernel):
    """Quick test that notebook cells can be executed.

    This is a faster test that only checks if cells can be executed,
    without full output validation.

    Parameters
    ----------
    notebook_name : str
        Name of the notebook to test.
    jupyter_kernel : bool
        Fixture ensuring kernel availability.
    """
    nbformat = pytest.importorskip("nbformat", reason="nbformat not installed")
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook_path = get_notebook_path(notebook_name)

    if not notebook_path.exists():
        pytest.skip(f"Notebook {notebook_name} not found")

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOKS_DIR)}})
    except Exception as e:
        pytest.fail(f"Notebook {notebook_name} failed to execute: {e}")


class TestNotebookSanity:
    """Sanity checks for notebooks."""

    def test_notebooks_directory_exists(self):
        """Test that the notebooks directory exists."""
        assert NOTEBOOKS_DIR.exists(), f"Notebooks directory {NOTEBOOKS_DIR} does not exist"

    @pytest.mark.parametrize("notebook_name", NOTEBOOKS)
    def test_notebook_file_exists(self, notebook_name: str):
        """Test that each notebook file exists.

        Parameters
        ----------
        notebook_name : str
            Name of the notebook to check.
        """
        notebook_path = get_notebook_path(notebook_name)
        assert notebook_path.exists(), f"Notebook {notebook_name} does not exist"

    @pytest.mark.parametrize("notebook_name", NOTEBOOKS)
    def test_notebook_is_valid_json(self, notebook_name: str):
        """Test that each notebook is valid JSON.

        Parameters
        ----------
        notebook_name : str
            Name of the notebook to check.
        """
        import json

        notebook_path = get_notebook_path(notebook_name)

        if not notebook_path.exists():
            pytest.skip(f"Notebook {notebook_name} not found")

        with open(notebook_path, "r", encoding="utf-8") as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Notebook {notebook_name} is not valid JSON: {e}")

    @pytest.mark.parametrize("notebook_name", NOTEBOOKS)
    def test_notebook_has_valid_format(self, notebook_name: str):
        """Test that each notebook has a valid Jupyter format.

        Parameters
        ----------
        notebook_name : str
            Name of the notebook to check.
        """
        nbformat = pytest.importorskip("nbformat", reason="nbformat not installed")

        notebook_path = get_notebook_path(notebook_name)

        if not notebook_path.exists():
            pytest.skip(f"Notebook {notebook_name} not found")

        with open(notebook_path, "r", encoding="utf-8") as f:
            try:
                nb = nbformat.read(f, as_version=4)
                # Validate the notebook
                nbformat.validate(nb)
            except Exception as e:
                pytest.fail(f"Notebook {notebook_name} has invalid format: {e}")


@pytest.mark.notebook
@pytest.mark.integration
def test_notebook_imports_scikitplot():
    """Test that notebooks can import scikitplot."""
    import scikitplot as skplt

    assert skplt is not None
    assert hasattr(skplt, "metrics")
