"""
Configuration for the API reference documentation.
"""

def _get_guide(*refs, is_developer=False):
    """Get the rst to refer to user/developer guide.

    `refs` is several references that can be used in the :ref:`...` directive.
    """
    if len(refs) == 1:
        ref_desc = f":ref:`{refs[0]}` section"
    elif len(refs) == 2:
        ref_desc = f":ref:`{refs[0]}` and :ref:`{refs[1]}` sections"
    else:
        ref_desc = ", ".join(f":ref:`{ref}`" for ref in refs[:-1])
        ref_desc += f", and :ref:`{refs[-1]}` sections"

    guide_name = "Developer" if is_developer else "User"
    return f"**{guide_name} guide.** See the {ref_desc} for further details."


def _get_submodule(module_name, submodule_name):
    """Get the submodule docstring and automatically add the hook.

    `module_name` is e.g. `sklearn.feature_extraction`, and `submodule_name` is e.g.
    `image`, so we get the docstring and hook for `sklearn.feature_extraction.image`
    submodule. `module_name` is used to reset the current module because autosummary
    automatically changes the current module.
    """
    lines = [
        f".. automodule:: {module_name}.{submodule_name}",
        f".. currentmodule:: {module_name}",
    ]
    return "\n\n".join(lines)


"""
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to a dictionary that consists of the following
components:

short_summary (required)
    The text to be printed on the index page; it has nothing to do the API reference
    page of each module.
description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

Essentially, the rendered page would look like the following:

|---------------------------------------------------------------------------------|
|     {{ module_name }}                                                           |
|     =================                                                           |
|     {{ module_docstring }}                                                      |
|     {{ description }}                                                           |
|                                                                                 |
|     {{ section_title_1 }}   <-------------- Optional if one wants the first     |
|     ---------------------                   section to directly follow          |
|     {{ section_description_1 }}             without a second-level heading.     |
|     {{ section_autosummary_1 }}                                                 |
|                                                                                 |
|     {{ section_title_2 }}                                                       |
|     ---------------------                                                       |
|     {{ section_description_2 }}                                                 |
|     {{ section_autosummary_2 }}                                                 |
|                                                                                 |
|     More sections...                                                            |
|---------------------------------------------------------------------------------|

Hooks will be automatically generated for each module and each section. For a module,
e.g., `sklearn.feature_extraction`, the hook would be `feature_extraction_ref`; for a
section, e.g., "From text" under `sklearn.feature_extraction`, the hook would be
`feature_extraction_ref-from-text`. However, note that a better way is to refer using
the :mod: directive, e.g., :mod:`sklearn.feature_extraction` for the module and
:mod:`sklearn.feature_extraction.text` for the section. Only in case that a section
is not a particular submodule does the hook become useful, e.g., the "Loaders" section
under `sklearn.datasets`.
"""

API_REFERENCE = {
    "scikitplot": {
        "short_summary": "Settings and information visualization tools.",
        "description": None,
        "sections": [
            {
                "title": "Configure global settings and get information about the working environment.",
                "autosummary": [
                    "show_config",
                    "show_versions",
                ],
            },
        ],
    },
    "scikitplot._numcpp_api": {
        "short_summary": "NumCpp Api functions and utilities by Pybind11 and Cython",
        "description": (
            _get_guide("numcpp_api")
        ),
        "sections": [
            {
                "title": "Pybind11 Functions Demo",
                "description": _get_submodule("scikitplot._numcpp_api", "py_numcpp_api"),
                "autosummary": [
                    "py_numcpp_api.py_get_version",
                    "py_numcpp_api.py_print_message",
                    "py_numcpp_api.py_random_array",
                    "py_numcpp_api.py_sum_of_squares",
                ],
            },
            {
                "title": "Cython Functions Demo",
                "description": _get_submodule("scikitplot._numcpp_api", "cy_numcpp_api"),
                "autosummary": [
                    "cy_numcpp_api.py_get_version",
                    "cy_numcpp_api.py_print_message",
                    "cy_numcpp_api.py_say_hello_inline",
                    "cy_numcpp_api.py_random_array",
                    "cy_numcpp_api.py_sum_of_squares",
                ],
            },
        ],
    },
    "scikitplot.api": {
        "short_summary": "Functional Api for Visualizations (function-based, implicit)",
        "description": (
            _get_guide("api")
        ),
        "sections": [
            {
                "title": "Plot a PCA representation",
                "description": (
                    _get_submodule("scikitplot.api", "decomposition")
                    + "\n\n"
                    + _get_guide("decomposition")
                ),
                "autosummary": [
                    "decomposition.plot_pca_2d_projection",
                    "decomposition.plot_pca_component_variance",
                ],
            },
            {
                "title": "Plot Estimators (model) object instances",
                "description": (
                    _get_submodule("scikitplot.api", "estimators")
                    + "\n\n"
                    + _get_guide("estimators")
                ),
                "autosummary": [
                    # Regressor estimators
                    # Classifier estimators
                    "estimators.plot_feature_importances",
                    # Classifier models scalability
                    "estimators.plot_learning_curve",
                    # Cluster estimators
                    "estimators.plot_elbow",
                ],
            },
            {
                "title": "Experimental functions",
                "description": (
                    _get_submodule("scikitplot.api", "experimental")
                    + "\n\n"
                    + _get_guide("experimental")
                ),
                "autosummary": [
                    "experimental.expit",
                    "experimental.softmax",
                ],
            },
            {
                "title": "KeyToDataScience",
                "description": (
                    _get_submodule("scikitplot.api", "kds")
                    + "\n\n"
                    + _get_guide("kds")
                ),
                "autosummary": [
                    "kds.print_labels",
                    "kds.decile_table",
                    "kds.plot_cumulative_gain",
                    "kds.plot_lift",
                    "kds.plot_lift_decile_wise",
                    "kds.plot_ks_statistic",
                    "kds.report",
                ],
            },
            {
                "title": "Plot model evaluation metrics",
                "description": (
                    _get_submodule("scikitplot.api", "metrics")
                    + "\n\n"
                    + _get_guide("metrics")
                ),
                "autosummary": [
                    # Regression metrics
                    "metrics.plot_residuals_distribution",
                    # Classification metrics
                    "metrics.plot_classifier_eval",
                    "metrics.plot_confusion_matrix",
                    "metrics.plot_precision_recall",
                    "metrics.plot_roc",
                    # Classification models scalability
                    "metrics.plot_calibration",
                    # Clustering metrics
                    "metrics.plot_silhouette",
                ],
            },
            {
                "title": "Predictive model insights",
                "description": (
                    _get_submodule("scikitplot.api", "modelplotpy")
                    + "\n\n"
                    + _get_guide("modelplotpy", "modelplotpy_financial")
                ),
                "autosummary": [
                    # Initialize modelplotpy object
                    "modelplotpy.ModelPlotPy",  # ModelPlotPy
                    # Gains, Lift and (cumulative) Response plots
                    "modelplotpy.plot_response",
                    "modelplotpy.plot_cumresponse",
                    "modelplotpy.plot_cumlift",
                    "modelplotpy.plot_cumgains",
                    "modelplotpy.plot_all",
                    # Business-savvy financial plots
                    "modelplotpy.plot_costsrevs",
                    "modelplotpy.plot_profit",
                    "modelplotpy.plot_roi",
                ],
            },
        ],
    },
    "scikitplot.probscale": {
        "short_summary": "Real probability scales for matplotlib.",
        "description": (
            _get_guide("probscale")
        ),
        "sections": [
            {
                "title": "probscale",
                "autosummary": [
                    'ProbScale',
                    'probplot',
                    'plot_pos',
                    'fit_line',
                ],
            },
        ],
    },
    "scikitplot.stats": {
        "short_summary": "Statistical tools for data visualization and interpretation.",
        "description": (
            _get_guide("stats")
        ),
        "sections": [
            {
                "title": "Discrete distributions",
                "description": _get_submodule("scikitplot.stats", "_tweedie"),
                "autosummary": [
                    '_tweedie.tweedie_gen',
                    '_tweedie.tweedie',
                ],
            },
        ],
    },
    "scikitplot.rcmod": {
        "short_summary": "Themeing.",
        "description": (
            _get_guide("rcmod")
        ),
        "sections": [
            {
                "title": "Themeing",
                "autosummary": [
                    'reset_defaults',
                    'reset_orig',
                ],
            },
        ],
    },
    "scikitplot.utils": {
        "short_summary": "Development Utilities",
        "description": (
            _get_guide("developers-guide-index", is_developer=True)
        ),
        "sections": [
            {
                "title": "Optimal mathematical operations",
                "description": _get_submodule("scikitplot.utils", "_helpers"),
                "autosummary": [
                    '_helpers.validate_labels',
                    '_helpers.cumulative_gain_curve',
                    '_helpers.binary_ks_curve',
                ],
            },
            {
                "title": "Input and parameter validation",
                "description": _get_submodule("scikitplot.utils", "validation"),
                "autosummary": [
                    'validation.validate_plotting_kwargs',
                ],
            },
            {
                "title": "Optimal matplotlib operations",
                "description": _get_submodule("scikitplot.utils", "_figures"),
                "autosummary": [
                    '_figures.combine_and_save_figures',
                ],
            },
        ],
    },
}


"""
CONFIGURING DEPRECATED_API_REFERENCE
====================================

DEPRECATED_API_REFERENCE maps each deprecation target version to a corresponding
autosummary block. It will be placed at the bottom of the API index page under the
"Recently deprecated" section. Essentially, the rendered section would look like the
following:

|------------------------------------------|
|     To be removed in {{ version_1 }}     |
|     --------------------------------     |
|     {{ autosummary_1 }}                  |
|                                          |
|     To be removed in {{ version_2 }}     |
|     --------------------------------     |
|     {{ autosummary_2 }}                  |
|                                          |
|     More versions...                     |
|------------------------------------------|

Note that the autosummary here assumes that the current module is `sklearn`, i.e., if
`sklearn.utils.Memory` is deprecated, one should put `utils.Memory` in the "entries"
slot of the autosummary block.

Example:

DEPRECATED_API_REFERENCE = {
    "0.24": [
        "model_selection.fit_grid_point",
        "utils.safe_indexing",
    ],
}
"""

DEPRECATED_API_REFERENCE = {
    "0.5": [
        "_factory_api",  # Visualizations (object-based, explicit)
        "api.metrics.plot_roc_curve",
        "api.metrics.plot_precision_recall_curve",
    ]
}  # type: ignore