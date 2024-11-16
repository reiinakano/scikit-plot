"""
scikit-plots Functional API

An intuitive library to add plotting functionality to scikit-learn objects

Documentation is available in the docstrings and
online at https://scikit-plots.github.io.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

######################################################################
## scikit-plots Functional API modules
######################################################################

# Modules
from . import (
  decomposition,
  estimators,
  experimental,  # C/Cpp-based Modules
  kds,
  metrics,
  modelplotpy,
)
# Deprecated namespaces, to be removed in v0.5.0
from .metrics import plotters  # noqa: F401,F403