"""
NumCpp C-API Python Extension Module

This module provides Python bindings for the NumCpp library, a C++ scientific computing library 
offering matrix operations, linear algebra, and other numerical tools. It exposes the core 
functionality of NumCpp via a C-API, enabling seamless integration of high-performance 
computations in Python applications, making it suitable for scientific computing, 
data analysis, and machine learning tasks.

Features
--------
- Matrix operations: Addition, multiplication, inversion, and more.
- Linear algebra: Eigenvalues, decompositions, and other routines.
- Random number generation: Tools for generating and analyzing random data.
- Multidimensional arrays: Support for n-dimensional data structures.
- High performance: Native C++ backend ensures speed and efficiency.

Import the module in your Python script and access NumCpp's functions directly:

Examples
--------
>>> from scikitplot import _numcpp_api as numcpp
>>> a = [[1, 2], [3, 4]]
>>> b = [[5, 6], [7, 8]]
>>> result = numcpp.matmul(a, b)
>>> print(result)
[[19, 22], [43, 50]]

Installation
------------
Ensure that the NumCpp library is installed and properly configured. The C++ library must 
be compiled, and the Python bindings should be built using the appropriate tools.

Refer to the NumCpp documentation for build instructions.

References
----------
- NumCpp GitHub Repository: https://github.com/dpilger26/NumCpp

Notes
-----
This module assumes familiarity with NumPy-style operations. Although the API is similar, 
NumCpp offers additional performance improvements via C++.

"""
# scikitplot/_numcpp_api/__init__.py

from . import (
  py_numcpp_api,
  cy_numcpp_api,
)

# Call the function to get the version
__version__ = py_numcpp_api.py_get_version()
__author__ = "David Pilger et al."
__author_email__ = "dpilger26@gmail.com"

# Define __all__ to control what gets imported with 'from module import *'
# Combine global names (explicitly defined in the module) and dynamically available names
__all__ = [
  'py_numcpp_api',
  'cy_numcpp_api',
]