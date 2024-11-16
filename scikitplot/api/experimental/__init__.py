"""
Experimental functions and utilities by Python and Cython.

This package contains experimental functions and utilities for use with 
Cython and Python implementations. The functions provided here
are optimized for performance and are implemented using Cython
to leverage its speed advantages.

This package is intended for experimental use. Functionality and APIs may 
change in future releases.
"""
# scikitplot/api/experimental/__init__.py

# Your package/module initialization code goes here
from ._logsumexp import *

# Optionally import other modules if available
try:
    from ._cy_experimental import *
except ImportError:
    # Optional modules not available
    pass