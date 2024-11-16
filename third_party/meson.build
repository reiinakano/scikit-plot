## scikit-plots/third_party/meson.build

######################################################################
## Submodule array-api-compat
######################################################################

if not fs.exists('array-api-compat/README.md')
  error(
  'Missing the `array-api-compat` submodule! ' + 
  'Run `git submodule update --init` to fix this.')
endif

# `array-api-compat` install to simplify import path;
# should be updated whenever new files are added to `array_api_compat`
# Copy all the .py files to the install dir, rather than using
# py.install_sources and needing to list them explicitely one by one
install_subdir(
  'array-api-compat/array_api_compat',
  install_dir: py.get_install_dir()/'scikitplot/_xp_core_lib',
  install_tag: 'runtime',
)
# Optionally add a message to confirm the installation
message(
  '\nArray API compatibility files installed successfully: ' +
  '@0@'.format(py.get_install_dir()/'scikitplot/_xp_core_lib')
)

######################################################################
## Submodule NumCpp
######################################################################

if not fs.exists('NumCpp/README.md')
  error('Missing the `NumCpp` submodule! ' +
  'Run `git submodule update --init` to fix this.')
endif

######################################################################
## include the headers directories
######################################################################

# include directories for Specific Subfolders
_numcpp_inc_dir = [
  'NumCpp/include/NumCpp/Core/Internal',
  'NumCpp/include',
  # 'NumCpp/include/NumCpp',
  # 'NumCpp/include/NumCpp/Coordinates',
  # 'NumCpp/include/NumCpp/Coordinates/ReferenceFrames',
  # 'NumCpp/include/NumCpp/Coordinates/Transforms',
  # 'NumCpp/include/NumCpp/Core',
  # 'NumCpp/include/NumCpp/Core/Internal',
  # 'NumCpp/include/NumCpp/DateTime',
  # 'NumCpp/include/NumCpp/Filter',
  # 'NumCpp/include/NumCpp/Filter/Boundaries',
  # 'NumCpp/include/NumCpp/Filter/Boundaries/Boundaries1d',
  # 'NumCpp/include/NumCpp/Filter/Boundaries/Boundaries2d',
  # 'NumCpp/include/NumCpp/Filter/Filters',
  # 'NumCpp/include/NumCpp/Filter/Filters/Filters1d',
  # 'NumCpp/include/NumCpp/Filter/Filters/Filters2d',
  # 'NumCpp/include/NumCpp/Functions',
  # 'NumCpp/include/NumCpp/ImageProcessing',
  # 'NumCpp/include/NumCpp/Integrate',
  # 'NumCpp/include/NumCpp/Linalg',
  # 'NumCpp/include/NumCpp/Linalg/svd',
  # 'NumCpp/include/NumCpp/Logging',
  # 'NumCpp/include/NumCpp/NdArray',
  # 'NumCpp/include/NumCpp/Polynomial',
  # 'NumCpp/include/NumCpp/PythonInterface',
  # 'NumCpp/include/NumCpp/Random',
  # 'NumCpp/include/NumCpp/Roots',
  # 'NumCpp/include/NumCpp/Rotations',
  # 'NumCpp/include/NumCpp/Special',
  # 'NumCpp/include/NumCpp/Utils',
  # 'NumCpp/include/NumCpp/Vector',
]

######################################################################
## Include header-only NumCpp
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_numcpp = include_directories(_numcpp_inc_dir)

# Optionally add a message to confirm the installation
message(
  '\nNumCpp Header compatibility files defined successfully: ' +
  '@0@'.format('third_party/NumCpp')
)

######################################################################
## NumCpp is a header-only library, which means it does not have any precompiled binary
## (like .so, .dll, or .a). 
######################################################################

# # Static library with C++ source file implementing bindings
# _st_lib_numcpp = static_library('_numcpp_lib_stat', [],
#   include_directories: inc_dir_numcpp,
#   dependencies: [],
#   link_with: [],
#   link_args: [],
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'hidden',
#   install: true,
#   install_dir: 'scikitplot/third_party',
# )
# # Shared (dynamic) library with C++ source file implementing bindings
# _dyn_lib_numcpp = library('_dyn_lib_numcpp', [],
#   include_directories: inc_dir_numcpp,
#   dependencies: [],
#   # link_with: [_st_lib_numcpp],  # Link with the static library
#   # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'hidden',
#   install: true,
#   install_dir: 'scikitplot/third_party',
# )

_global_numcpp_args = []
# Get the NumCpp Dependencies with/without library
dep_numcpp = declare_dependency(
  compile_args: _global_numcpp_args,
  dependencies: [],
  include_directories: inc_dir_numcpp,
  # link_with: [_numcpp_lib_dyn],  # Link with the static/shared library
  # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
)
# Optionally add a message to confirm the installation
message(
  '\nNumCpp Header dependency defined successfully: ' +
  '@0@ '.format('dep_numcpp')
)

######################################################################
## Submodule xla
######################################################################

if not fs.exists('xla/README.md')
  error('Missing the `xla` submodule! Run `git submodule update --init` to fix this.')
endif
# Optionally add a message to confirm the installation
message(
  '\nXLA compatibility files installed successfully: ' +
  '@0@'.format('third_party/xla')
)

######################################################################
## Submodule
######################################################################