"""
Utilities for building docs.
"""
from sphinx_gallery.notebook import add_code_cell, add_markdown_cell
try:
    import sklearn
    default_global_config = sklearn.get_config()

    # https://github.com/sphinx-gallery/sphinx-gallery/blob/master/sphinx_gallery/scrapers.py#L562
    def _reset_sklearn (gallery_conf, fname):
        """Reset sklearn config to default values."""
        sklearn.set_config(**default_global_config)
except Exception:
    pass


def notebook_modification_function(notebook_content, notebook_filename):
    """Implement JupyterLite-specific modifications of notebooks."""
    notebook_content_str = str(notebook_content)
    warning_template = "\n".join(
        [
            "<div class='alert alert-{message_class}'>",
            "",
            "# JupyterLite warning",
            "",
            "{message}",
            "</div>",
        ]
    )

    message_class = "warning"
    message = (
        "Running the scikit-plots examples in JupyterLite is experimental and you may"
        " encounter some unexpected behavior.\n\nThe main difference is that imports"
        " will take a lot longer than usual, for example the first `import scikitplot`"
        "can take roughly 10-20s.\n\nIf you notice problems, feel free to open an"
        " [issue](https://github.com/scikit-plots/scikit-plots/issues/new/choose)"
        " about it."
    )

    markdown = warning_template.format(message_class=message_class, message=message)

    dummy_notebook_content = {"cells": []}
    add_markdown_cell(dummy_notebook_content, markdown)

    code_lines = []

    if "seaborn" in notebook_content_str:
        code_lines.append("%pip install seaborn")
    if "plotly.express" in notebook_content_str:
        code_lines.append("%pip install plotly")
    if "skimage" in notebook_content_str:
        code_lines.append("%pip install scikit-image")
    if "polars" in notebook_content_str:
        code_lines.append("%pip install polars")
    if "fetch_" in notebook_content_str:
        code_lines.extend(
            [
                "%pip install pyodide-http",
                "import pyodide_http",
                "pyodide_http.patch_all()",
            ]
        )
    # always import matplotlib and pandas to avoid Pyodide limitation with
    # imports inside functions
    code_lines.extend(["import matplotlib", "import pandas"])

    if code_lines:
        code_lines = ["# JupyterLite-specific code"] + code_lines
        code = "\n".join(code_lines)
        add_code_cell(dummy_notebook_content, code)

    notebook_content["cells"] = (
        dummy_notebook_content["cells"] + notebook_content["cells"]
    )


def reset_others(gallery_conf, fname):
    """Reset plotting functions."""
    # sklearn
    try:
        import sklearn        
    except Exception:
        pass
    else:
        _reset_sklearn(gallery_conf, fname)
    # plotly
    try:
        import plotly.io
    except Exception:
        pass
    else:
        plotly.io.renderers.default = "sphinx_gallery"
    # pyvista
    try:
        import pyvista
    except Exception:
        pass
    else:
        pyvista.OFF_SCREEN = True
        # Preferred plotting style for documentation
        pyvista.set_plot_theme("document")
        pyvista.global_theme.window_size = [1024, 768]
        pyvista.global_theme.font.size = 22
        pyvista.global_theme.font.label_size = 22
        pyvista.global_theme.font.title_size = 22
        pyvista.global_theme.return_cpos = False
        # necessary when building the sphinx gallery
        pyvista.BUILDING_GALLERY = True
        pyvista.set_jupyter_backend(None)