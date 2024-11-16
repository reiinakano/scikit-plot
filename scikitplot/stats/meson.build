## scikit-plots/scikitplot/stats/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_stats_cython_tree = [
  _cython_tree,
  fs.copyfile('__init__.py'),
]

######################################################################
## Next
######################################################################

subdir('_astropy')