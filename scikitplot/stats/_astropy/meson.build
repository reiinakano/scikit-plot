## scikit-plots/scikitplot/stats/_astropy/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_astropy_cython_tree = [
  _stats_cython_tree,
  fs.copyfile('__init__.py'),
]

######################################################################
## Next
######################################################################

subdir('stats')