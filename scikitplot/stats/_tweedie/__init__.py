"""
tweedie is a Python library implementing scipy's rv_continuous class for the Tweedie family.
The Tweedie family is a member of the exponential dispersion model family
and is commonly used in the insurance indsutry to model claim amounts
for insurance policies (exposure).

The main focus of this package is the compound-Poisson behavior,
specifically where 1 < p < 2. However, it should be possible to calculate
the distribution for all the possible values of p.
"""
# scikitplot/stats/_tweedie/__init__.py

from ._tweedie_dist import *

__version__ = "0.0.9.dev0"
__author__ = "Peter Quackenbush"
__author_email__ = "pquack@gmail.com"

# Define __all__ to control what gets imported with 'from module import *'
# Combine global names (explicitly defined in the module) and dynamically available names
__all__ = [
  'tweedie_gen',
  'tweedie',
]