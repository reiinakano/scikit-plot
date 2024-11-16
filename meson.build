## scikit-plots/meson.build

######################################################################
## Project
######################################################################

# Project configuration
project(
  'scikit-plots',
  'c', 'cpp', 'cython',
  # Note that the git commit hash cannot be added dynamically here
  # That only happens when importing from a git repository.
  # See `scikitplot/__init__.py`
  version: run_command(
    ['python', 'scikitplot/_build_utils/version.py'],
    check: true,
  ).stdout().strip(),
  license: 'BSD-3 Clause',
  meson_version: '>= 1.1.0',
  default_options: [
    # 'pkgconfig.relocatable=true',
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'fortran_std=legacy',            # Set C standard to C11
    'c_std=c11',                     # Set C standard to C11
    'cpp_std=c++17',                 # Set C++ standard to C++17
    # https://mesonbuild.com/Builtin-options.html#details-for-warning_level
    # 'warning_level=3',             # You can specify warning levels here if needed
    # 'werror=true',                 # Treat warnings as errors globally
  ],
)

# Get the directory relative to the top-level project directory
# project_root = meson.project_source_root()
# 
# Get the directory relative to the current meson.build file
# current_dir = meson.current_source_dir()
# 
# Get the directory relative to the build directory
# build_dir = meson.current_build_dir()

######################################################################
## fortran
######################################################################

# Conditional Fortran setup
if get_option('fortran').enabled()
  # Adding at project level causes many spurious -lgfortran flags.
  add_languages('fortran', native: false)
  ff = meson.get_compiler('fortran')
  if ff.has_argument('-Wno-conversion')
    add_project_arguments(
      '-Wno-conversion',
      language: ['fortran'],
    )
  endif
endif

######################################################################
## Fortran Global compiler flags/options argument
######################################################################

# Fortran compiler options
# The default list of warnings to ignore from Fortran code. There is a lot of
# old, vendored code that is very bad and we want to compile it silently (at
# least with GCC and Clang)
_global_fortran_args = [
  # Define warning flags for Fortran compilation
  '-Wno-argument-mismatch',      # Suppress warnings about mismatched argument types in subroutine/function calls
  '-Wno-conversion',             # Suppress warnings about implicit type conversions
  '-Wno-intrinsic-shadow',       # Suppress warnings about shadowing intrinsic names
  '-Wno-maybe-uninitialized',    # Suppress warnings about variables that might be used uninitialized
  '-Wno-surprising',             # Suppress warnings about non-standard Fortran practices
  '-Wno-uninitialized',          # Suppress warnings about uninitialized variables
  '-Wno-unused-dummy-argument',  # Suppress warnings about dummy arguments that are unused
  '-Wno-unused-label',           # Suppress warnings about unused labels
  '-Wno-unused-variable',        # Suppress warnings about unused variables
  '-Wno-tabs',                   # Suppress warnings about tabs in the source code (often used in legacy code)
]

# Conditional Fortran setup
if get_option('fortran').enabled()
  # Compiler options for Fortran
  _global_fortran_args = ff.get_supported_arguments(
    _global_fortran_args
  )
  add_project_arguments(
    _global_fortran_args,
    language: ['fortran'],
  )
endif

######################################################################
## C and C++
######################################################################

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=9.1')
    error('scikit-plots requires GCC >= 9.1')
  endif
elif cc.get_id() == 'clang' or cc.get_id() == 'clang-cl'
  if not cc.version().version_compare('>=12.0')
    error('scikit-plots requires clang >= 12.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error(
      'scikit-plots requires at least vc142 (default with Visual Studio 2019) ' +
      'when building with MSVC'
    )
  endif
endif

######################################################################
## C strict aliasing
######################################################################

# Add default compile flags for any compiler that supports them.
# Note that MSVC does not support strict aliasing at all, and neither do the
# Intel compilers on Windows, so the `-fno` flavor of the flag should be fine.
add_project_arguments(
  cc.get_supported_arguments('-fno-strict-aliasing'),
  language : ['c'],
)

######################################################################
## Clang C and C++ strict behavior
######################################################################

# Clang defaults to a non-strict floating error point model, but we need strict
# behavior. `-ftrapping-math` is equivalent to `-ffp-exception-behavior=strict`.
# This flag is also required to prevent the activation of SIMD partial load workarounds.
# For further clarification, refer to gh-24461.
cc_id = cc.get_id()
if cc_id.startswith('clang')
  # Determine the compiler flags for trapping math exceptions.
  trapping_math = {
    'clang-cl': '/clang:-ftrapping-math',
  }.get(cc_id, '-ftrapping-math')
  # Check if the compiler supports the trapping math flag.
  if cc.has_argument(trapping_math)
    # TODO: Consider upgrading the vendored Meson to 1.3.0 to support the parameter `werror`
    # Detect whether the compiler actually supports strict handling of floating-point exceptions
    # by treating warnings as errors.
    if cc.compiles('int main() { return 0; }', args: [trapping_math, '-Werror'])
      trapping_math = [trapping_math, '-DNPY_HAVE_CLANG_FPSTRICT']
    else
      # Suppress warnings about unsupported floating-point optimization.
      trapping_math = [trapping_math, '-Wno-unsupported-floating-point-opt']
      # Inform the user about the workaround.
      message(
        'NumPy is being built against a version of Clang that does not strictly enforce ' +
        'floating-point exception handling. Workarounds will be used, which may impact performance.\n' +
        'Consider upgrading Clang to the latest version.'
      )
    endif
    add_project_arguments(
      trapping_math,
      language: ['c', 'cpp']
    )
  endif
endif

######################################################################
## C and C++ Global compiler flags/options argument
######################################################################

# Global compiler options
_global_args = [
  '-DDEBUG',                        # Define DEBUG macro to enable debug mode
  '-W',                             # Enable all warnings
  '-Werror=unused-variable',        # Treat unused variables as errors to enforce code quality
  '-Wno-cast-align',                # Suppress warnings if a pointer is cast to a type with different alignment
  '-Wno-cast-qual',                 # Suppress warnings if a pointer is cast to a type with different qualifiers
  '-Wno-conversion',                # Suppress warnings about type conversions
  '-Wno-double-promotion',          # Suppress warnings if a float is promoted to double
  '-Wno-error=unused-function',     # Suppress errors for unused functions
  '-Wno-float-equal',               # Suppress warnings if floating-point values are compared for equality
  '-Wno-format-truncation',         # Suppress warnings about format string truncations
  '-Wno-misleading-indentation',    # Suppress warnings about misleading indentation
  '-Wno-parentheses',               # Suppress warnings about unnecessary parentheses
  '-Wno-pointer-arith',             # Suppress warnings about pointer arithmetic
  '-Wno-redundant-decls',           # Suppress warnings about redundant declarations
  '-Wno-sequence-point',            # Suppress warnings about potential sequence point issues
  '-Wno-shadow',                    # Suppress warnings if a local variable shadows another variable
  '-Wno-sign-compare',              # Suppress warnings about comparisons between signed and unsigned integers
  '-Wno-switch',                    # Suppress warnings about switch statement cases that are not handled
  '-Wno-undef',                     # Suppress warnings if an undefined macro is used
  '-Wno-unused-but-set-parameter',  # Suppress warnings about parameters that are set but not used
  '-Wno-unused-function',           # Suppress warnings about unused functions
  '-Wno-unused-label',              # Suppress warnings about unused labels
  '-Wno-unused-parameter',          # Suppress warnings about unused parameters
  '-Wno-unused-result',             # Suppress warnings about unused results
  '-Wno-unused-variable',           # Suppress warnings about unused variables
  '-Wno-write-strings',             # Treat string literals as `const char[]`
]
# Compiler options for C
_global_c_args = cc.get_supported_arguments(
  _global_args + [
    '-Wno-implicit-function-declaration',  # Suppress warnings about implicit function declarations
  ]
)
# Compiler options for C++
_global_cpp_args = cpp.get_supported_arguments(
  _global_args + [
    '-pthread',                  # Add pthread support for POSIX systems
    '-Wno-class-memaccess',      # Suppress warnings about accessing class members inappropriately
    '-Wno-cpp',                  # Suppress warnings about C++ preprocessor directives
    '-Wno-int-in-bool-context',  # Suppress warnings about int used in boolean context
    '-Wno-non-virtual-dtor',     # Suppress warnings about non-virtual destructors
    '-Wno-pedantic',             # Suppress strict ISO C++ compliance warnings
    '-Wno-terminate',            # Suppress warnings about code that might terminate unexpectedly
  ]
)

# Check for std::thread support and add appropriate flags
if cpp.has_function('std::thread')
  # Enable multithreading with STL parallel execution policies
  _global_cpp_args += ['-DNUMCPP_USE_MULTITHREAD']
endif

add_project_arguments(
  _global_c_args,
  language: ['c'],
)

add_project_arguments(
  _global_cpp_args,
  language: ['cpp'],
)

######################################################################
## C and C++ math functions, symbols
######################################################################

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for scikit-image). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it. For Fortran code, Meson already adds `-lm`.
dep_m = cc.find_library('m', required : false)
if dep_m.found()
  add_project_link_arguments(
    ['-lm'],
    language : ['c'],
  )
endif

# Hide symbols when building on Linux with GCC. For Python extension modules,
# we only need `PyInit_*` to be public, anything else may cause problems. So we
# use a linker script to avoid exporting those symbols (this is in addition to
# Meson using `-fvisibility=hidden` for C and `-fvisibility-inlines-hidden` for
# C++ code. See gh-15996 for details.
_linker_script = (
  meson.project_source_root() / 'scikitplot/_build_utils/link-version-pyinit.map'
)
version_link_args = ['-Wl,--version-script=' + _linker_script]
# Note that FreeBSD only accepts version scripts when -shared is passed,
# hence we need to pass that to `cc.links` explicitly (flag is already
# present for `extension_module` invocations).
if not cc.links(
    '',
    name: '-Wl,--version-script',
    args: ['-shared', version_link_args]
  )
  version_link_args = []
endif

######################################################################
## Platform
######################################################################

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'
# is_mingw = is_windows and cc.get_define('__MINGW32__') != ''

# Conditional Fortran setup
if get_option('fortran').enabled()
  if is_mingw and ff.get_id() != 'gcc'
    error(
      'If you are using GCC on Windows, you must also use GFortran! ' +
      'Detected: @0@'.format(ff.get_id())
    )
  endif
endif

# Emit a warning for 32-bit Python installs on Windows; users are getting
# unexpected from-source builds there because we no longer provide wheels.
if is_windows and py.has_variable('EXT_SUFFIX')
  if py.get_variable('EXT_SUFFIX').contains('win32')
    warning(
      'You are building from source on a 32-bit Python install. ' +
      'Scikitplot does not provide 32-bit wheels; install 64-bit Python ' +
      'if you are having issues!'
    )
  endif
endif

######################################################################
## macOS 14
######################################################################

if host_machine.system() == 'darwin'
  if cc.has_link_argument('-Wl,-ld_classic')
    # New linker introduced in macOS 14 not working yet, see gh-19357 and gh-19387
    add_project_link_arguments('-Wl,-ld_classic', language : ['c', 'cpp', 'fortran'])
  endif
  if cc.has_link_argument('-Wl,-dead_strip')
    # Allow linker to strip unused symbols
    add_project_link_arguments('-Wl,-dead_strip', language : ['c', 'cpp', 'fortran'])
  endif
endif

######################################################################
## python
######################################################################

# Import the Python module from Meson to interact with the Python installation
# https://mesonbuild.com/Python-module.html
mod_py = import('python')
# NOTE: with Meson >=0.64.0 we can add `pure: false` here and remove that line
# everywhere else, see https://github.com/mesonbuild/meson/pull/10783.
py = mod_py.find_installation(pure: false)

# Get the Python Dependencies
dep_py = py.dependency()

######################################################################
## python headers 'Python.h'
######################################################################

if not cc.has_header('Python.h', dependencies: dep_py)
  error('Cannot compile `Python.h`. ' +
    'Perhaps you need to install python-dev|python-devel')
endif

# # Attempt to get the Python include directory
# _py_inc_dir = meson.get_external_property(
#   'python-include-dir',
#   'not-given'
# )
# if _py_inc_dir == 'not-given'
#   _py_inc_dir = run_command(
#     py,
#     [
#       '-c',
#       '''\
# import sysconfig
# print(sysconfig.get_path("include"))
# '''
#     ],
#     check: true
#   ).stdout().strip()
# endif
# # Use the include directory in your build setup
# inc_dir_py = include_directories(_py_inc_dir)

# Print the include directory
# message('python include dir: @0@'.format(_py_inc_dir))

######################################################################
## pybind11 is a lightweight header-only library
## that exposes C++11 types in Python and vice versa
## https://github.com/pybind/pybind11
## https://pybind11.readthedocs.io/_/downloads/en/latest/pdf/
## pybind11/pybind11.h includes Python.h,
## as such it must be the first file included in any source file
## or header for the same reasons as 'Python.h'.
######################################################################

# pybind11 — Seamless operability between C++11 and Python
dep_pybind11 = dependency('pybind11', version: '>=2.13.1')

######################################################################
## pythran
######################################################################

pythran = find_program(
  'pythran',
  native: true,
  version: '>=0.14.0',
  required: get_option('pythran').enabled(),
)
if get_option('pythran').enabled()
  # xsimd is unvendored from pythran by conda-forge, and due to a compiler
  # activation bug the default <prefix>/include/ may not be visible (see
  # gh-15698). Hence look for xsimd explicitly.
  dep_xsimd = dependency('xsimd', required: false)
endif

######################################################################
## pythran args
######################################################################

# Pythran build flags
if get_option('pythran').enabled()
  # Base Pythran-specific flags
  _pythran_base_args = [
    '-DENABLE_PYTHON_MODULE',  # Enable Python module support
    '-D__PYTHRAN__=3',         # Define Pythran version
    '-DPYTHRAN_BLAS_NONE',     # Disable BLAS
  ]
  # Pythran base and C++ warning flags
  _global_pythran_cpp_args = _pythran_base_args + _global_cpp_args
  add_project_arguments(
    _global_pythran_cpp_args,
    language: ['pythran'],
  )
endif

######################################################################
## pythran include dir
######################################################################

# Pythran include directory
if get_option('pythran').enabled()
  _pythran_inc_dir = meson.get_external_property(
    'pythran-include-dir',
    'not-given'
  )
  if incdir_pythran == 'not-given'
    # If not specified, try to query Pythran from the build python
    _pythran_inc_dir = run_command(py,
      [
        '-c',
      '''\
import os; os.chdir("..")
import pythran
print(os.path.dirname(pythran.__file__))
'''
      ],
      check: true
    ).stdout().strip()
  endif  
  inc_dir_pythran = include_directories(_pythran_inc_dir)

  message(
    'pythran include dir: @0@'.format(_pythran_inc_dir)
  )
endif

######################################################################
## cython
######################################################################

# Find the Cython program, which is needed to compile .pyx files to .c or .cpp files
cy = meson.get_compiler('cython')

# generator() doesn't accept compilers, only found programs - cast it as cli_cython.
cli_cython = find_program(cy.cmd_array()[0])

# Scipy variant of Cython command
# cli_cython = find_program('scikitplot/_build_utils/cythoner.py')

######################################################################
## Common Cython arguments
######################################################################

# cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@', '--include-dir', '@BUILD_ROOT@', '@INPUT@']

# Define the arguments for the Cython command. These arguments will be passed
# to the Cython compiler when generating the C source files.
cython_c_args = [
  # '-O2',  # Optimization level
  # '-X boundscheck=' + boundscheck,      # Control bounds checking (True/False)
  '-X language_level=3',                # Ensure Cython targets Python3 like '-3'
  # Prevents using negative indexing, which can provide a speedup for array handling.
  '-X wraparound=False',                # Disable negative indexing (wraparound)
  # Skips checking for uninitialized variables, improving performance at the cost of safety.
  '-X initializedcheck=False',          # Disable check for uninitialized variables
  # Skips checking for None values, improving performance when you're sure that variables won’t be None.
  '-X nonecheck=False',                 # Disable None checks
  # Enables C-style division (ignores Python division safety), improving performance.
  '-X cdivision=True',                  # Enable C division
  # Disables profiling support, which improves speed if profiling isn’t needed.
  '-X profile=False',                   # Disable profiling
  '--fast-fail',                        # Halt on first error during compilation
  # Specifies the include directory where Cython can find the necessary headers.
  # Needed for cython imports across subpackages, e.g. cluster pyx that
  # cimports metrics pxd
  # '--include-dir', '@BUILD_ROOT@',      # Placeholder for Meson to replace later
  '--include-dir', meson.global_build_root(), # Directly resolves to the build root path
  # '--output-file', '@OUTPUT@',          # Specify output file for generated C code
  # '@INPUT@',                            # Input file placeholder managed by Meson
]
# If the Cython version is 3.1.0 or newer, add the option for free threading compatibility
if cy.version().version_compare('>=3.1.0')
  cython_c_args += ['-Xfreethreading_compatible=True']
endif
# Define additional arguments specifically for generating C++ files with Cython
cython_cpp_args = cython_c_args + ['--cplus']  # The '--cplus' flag tells Cython to generate C++ code

######################################################################
## Cython generator/compiler
## pyx -> c, pyx -> cpp generators, depending on copied pxd files and init
## https://mesonbuild.com/Reference-manual_functions.html#generator
## https://mesonbuild.com/Reference-manual_functions.html#custom_target
######################################################################

_gen_cython_args = [
  # '--include-dir', '@BUILD_ROOT@',      # Placeholder for Meson to replace later
  '--output-file', '@OUTPUT@',          # Specify output file for generated C code
  '@INPUT@',                            # Input file placeholder managed by Meson
]

# Create a generator for converting .pyx files to .c files using the defined Cython arguments
gen_cython_c = generator(
  cli_cython,
  arguments : _gen_cython_args + cython_c_args,  # Use the previously defined arguments
  output : '@BASENAME@.c',             # The output will be a .c file with the same base name as the input .pyx file
  # depends : _cython_tree             # Optional: Specify dependencies like .pxd files, if any
)
# Create a generator for converting .pyx files to .cpp files (C++ code) using Cython arguments
gen_cython_cpp = generator(
  cli_cython,
  arguments : _gen_cython_args + cython_cpp_args,  # Use the C++ specific arguments
  output : '@BASENAME@.cpp',         # The output will be a .cpp file with the same base name as the input .pyx file
  # depends : [_cython_tree]         # Optional: Specify dependencies like .pxd files, if any
)

# Conditional generation with Pythran, a tool that accelerates Python code.
# This block is only executed if the 'use_pythran' option is enabled.
if get_option('pythran').enabled()
  # Define a generator for Pythran to convert Python files into C++ code
  # Note: The arguments need to be updated when Pythran supports running without the GIL (Global Interpreter Lock)
  pythran_gen = generator(
    pythran,
    arguments : ['-E', '@INPUT@', '-o', '@OUTPUT@'],  # '-E' expands macros, '@INPUT@' is the input file, and '@OUTPUT@' is the output
    output : '@BASENAME@.cpp',                        # Output will be a .cpp file
  )
endif

######################################################################
## Set common build flags for C and C++ code
######################################################################

# Adapted from Scipy. mingw is untested and not officially supported. If you
# ever bump into issues when trying to compile for mingw, please open an issue
# in the scikit-learn issue tracker
cython_c_flags = [
  '-DCYTHON_CCOMPLEX=0',  # see gh-18975 for why we need this
]
if is_mingw
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  add_project_link_arguments(gcc_link_args, language: ['c', 'cpp', 'fortran'])
  # Force gcc to float64 long doubles for compatibility with MSVC
  # builds, for C only.
  add_project_arguments('-mlong-double-64', language: 'c')
  # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
  add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
  ###############################################################
  # Manual add of MS_WIN64 macro when not using MSVC.
  # https://bugs.python.org/issue28267
  bitness = run_command(
    '_build_utils/gcc_build_bitness.py',
    check: true
  ).stdout().strip()
  if bitness == '64'
    add_project_arguments(
      '-DMS_WIN64',
      language: ['c', 'cpp'],
    )
  endif
  ###############################################################
  # Silence warnings emitted by PyOS_snprintf for (%zd), see
  # https://github.com/rgommers/scipy/issues/118.
  # Use as c_args for extensions containing Cython code
  cython_c_flags += ['-Wno-format-extra-args', '-Wno-format']
  # Flag needed to work around BLAS and LAPACK Gfortran dependence on
  # undocumented C feature when passing single character string arguments. See:
  #   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90329
  #   https://github.com/wch/r-source/blob/838f9d5a7be08f2a8c08e47bcd28756f5d0aac90/src/gnuwin32/MkRules.rules#L121
  add_project_arguments(
    '-fno-optimize-sibling-calls',
    language: ['fortran']
  )
endif

# Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
# Cython doesn't always get this right itself (see, e.g., gh-16800), so
# explicitly add the define as a compiler flag for Cython-generated code.
if is_windows
  use_math_defines = ['-D_USE_MATH_DEFINES']
else
  use_math_defines = []
endif

# Define C and C++ specific arguments
cython_c_flags += [use_math_defines]


# Adapted from Numpy.
# Same as NPY_CXX_FLAGS (TODO: extend for what ccompiler_opt adds)
cython_cpp_flags = cython_c_flags + [
  '-D__STDC_VERSION__=0',  # for compatibility with C headers
]
# if cc.get_argument_syntax() != 'msvc'
#   cython_cpp_flags += [
#     '-fno-exceptions',  # no exception support
#     '-fno-rtti',        # no runtime type information
#   ]
# endif

######################################################################
## numpy include dir
######################################################################

# When cross-compiling scikitplots, the compiler needs access to NumPy and Pythran
# headers for the host platform (where the package will actually run). These
# headers may be incompatible with any corresponding headers that might be
# installed on the build system (where the compilation is performed). To make
# sure that the compiler finds the right headers, paths can be configured in
# the 'properties' section of a Meson cross file:
#
#   [properties]
#   numpy-include-dir = '/path/to/host/numpy/includes'
#   pythran-include-dir = '/path/to/host/pythran/includes'
#
# If a cross file is not provided or does not specify either of these
# properties, fall back to running Python on the build system to query NumPy or
# Pythran directly for the appropriate paths. This will detect appropriate
# paths for native builds. (This might even work for certain build/host cross
# combinations, but don't rely on that.)
#
# For more information about cross compilation in Meson, including a definition
# of "build" and "host" in this context, refer to
#
#  https://mesonbuild.com/Cross-compilation.html

# NumPy include directory - needed in all submodules
# The chdir is needed because within numpy there's an `import signal`
# statement, and we don't want that to pick up scipy's signal module rather
# than the stdlib module. The try-except is needed because when things are
# split across drives on Windows, there is no relative path and an exception
# gets raised. There may be other such cases, so add a catch-all and switch to
# an absolute path. Relative paths are needed when for example a virtualenv is
# placed inside the source tree; Meson rejects absolute paths to places inside
# the source tree.
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#   [properties]
#   numpy-include-dir = '/abspath/to/host-pythons/site-packages/numpy/core/include'
#
# This uses the path as is, and avoids running the interpreter.
_np_inc_dir = meson.get_external_property(
  'numpy-include-dir',
  'not-given',
)
if _np_inc_dir == 'not-given'
  _np_inc_dir = run_command(py,
    [
      '-c',
      '''\
import os
import numpy as np
try:
  # get absolute path
  _np_inc_dir = np.get_include()
  if not os.path.exists(_np_inc_dir):
    raise FileNotFoundError("")
except Exception as e:
  os.chdir('.')
  _np_inc_dir = np.get_include()
  # get relative path
  _np_inc_dir = os.path.relpath(_np_inc_dir)
print(_np_inc_dir)
'''
    ],
    check: true
  ).stdout().strip()

  # We do need an absolute path to feed to `cc.find_library` below
  _np_inc_dir_abs = run_command(py,
    [
      '-c',
      'import numpy; print(numpy.get_include())'
    ],
    check: true
  ).stdout().strip()
else
  _np_inc_dir_abs = _np_inc_dir
endif
message(
  '\nnumpy include dir    : @0@'.format(_np_inc_dir) +
  '\nnumpy include dir abs: @0@'.format(_np_inc_dir_abs)
)

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_np = include_directories(_np_inc_dir)

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation.
_global_np_args = [
  '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'
]
# Get the Numpy Dependencies
dep_np = declare_dependency(
  compile_args: _global_np_args,
  dependencies: [],
  include_directories: inc_dir_np,
)

######################################################################
## py copier
######################################################################

cli_copier = find_program(['cp', 'scikitplot/_build_utils/copyfiles.py'])

######################################################################
## py tempita
######################################################################

# Usage: custom_target(command: [py, cli_tempita, '@INPUT@', '-o', '@OUTDIR@'],
# Usage: custom_target(command: [cli_tempita, '@INPUT@', '-o', '@OUTPUT@'],
cli_tempita = find_program('scikitplot/_build_utils/tempita.py')

# Usage: tempita_gen.process('__config__.py.in')
# Used for templated py/C/Cpp/pyx/pxd code
gen_tempita = generator(cli_tempita,
  arguments : ['@INPUT@', '-o', '@OUTPUT@'],
  output : '@BASENAME@',
)

######################################################################
## Installation tags
## https://mesonbuild.com/Installing.html#installing
## https://mesonbuild.com/Installing.html#installation-tags
## 
## It is possible to install only a subset of the installable files using
## meson install --tags tag1,tag2 command line. When --tags is specified,
## only files that have been tagged with one of the tags are going to be installed.
######################################################################

######################################################################
## Copy all the .py files
## https://mesonbuild.com/Reference-manual_functions.html#install_subdir
## https://mesonbuild.com/Installing.html#installing
######################################################################

# Copy all the .py files to the install dir, rather than using
# py.install_sources and needing to list them explicitely one by one
install_subdir(
  'scikitplot',
  install_dir: py.get_install_dir(),  # Folder relative to site-packages to install to
  install_tag: 'runtime',
  # strip_directory : true,
  # exclude_directories: ['third_party', ],  # Include all directories if not needed
)
# Optionally add a message to confirm the installation
message(
  '\nscikitplot files installed successfully: ' +
  '@0@'.format(py.get_install_dir())
)

######################################################################
## Fs-module
## https://mesonbuild.com/Fs-module.html
######################################################################

# https://mesonbuild.com/Fs-module.html
fs = import('fs')

######################################################################
## Next
######################################################################

subdir('third_party')
subdir('scikitplot')