## scikit-plots/scikitplot/_experimental/meson.build

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# The Cython files that need to be processed, including dependencies and includes
# `_cython_tree` should have been defined earlier in the build script
# to make sure scikitplot/__init__.py is copied
experimental_cython_tree = [
  _api_cython_tree,                       # also _cython_tree
  fs.copyfile('__init__.py'),             # Ensure __init__.py is copied early
  fs.copyfile('_complexstuff.pxd'),       # Cython definition file for C/C++ code
  fs.copyfile('_cy_experimental.pxd'),    # Cython definition file for C/C++ code
]

######################################################################
## include the headers directories
######################################################################

# include_directories for Specific Subfolders
_experimental_inc_dir = [
  'src',
  'src/complexstuff',
]

######################################################################
## library
######################################################################

# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
# Specify Include directories where your headers are located
inc_dir_experimental = include_directories(_experimental_inc_dir)

# Static library
st_lib_experimental = static_library('st_lib_experimental',
  [
    'src/complexstuff/special_wrappers.cpp',  # C++ file    
  ],
  dependencies: [
    dep_pybind11,
    dep_np,
  ],
  include_directories: [
    inc_dir_np,
    inc_dir_experimental,
  ],
  link_with: [],
  # c_args: cython_c_flags,     # Apply C-specific flags for C files
  cpp_args: cython_cpp_flags,  # Apply C++-specific flags for C++ files
  gnu_symbol_visibility: 'hidden',
  # install: true,
)

######################################################################
## extension metadata
######################################################################

# Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
experimental_extension_metadata = {
  # Define the '.pyx' extension with C++ support
  '_cy_experimental':
    {
      'sources': ['_cy_experimental.pyx'],          # Source .pyx file
      'dependencies': [
        dep_pybind11,
        dep_np,
      ],                                            # External libraries and dependencies
      'include_directories': [
        inc_dir_np,
        inc_dir_experimental,
      ],                                            # Include dirs for compilation
      'link_with': [
        st_lib_experimental,
      ],                                            # Link with the created static library
      'override_options': ['cython_language=cpp'],  # Ensure Cython knows to generate C++ code
      # 'c_args': cython_c_flags,                     # Additional C/C++ arguments
      'cpp_args': cython_cpp_flags,
      'install': true,                              # Whether to install the .so file executable after building
      'subdir': 'scikitplot/api/experimental',      # Path where the module is located
    },
}


# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : experimental_extension_metadata
  py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + experimental_cython_tree,            # Sources and dependencies
    include_directories: ext_dict.get('include_directories', []),  # Include directories
    link_with: ext_dict.get('link_with', []),                      # Libraries to link with
    dependencies: ext_dict.get('dependencies', []),                # Additional dependencies if any
    override_options : ext_dict.get('override_options', []),       # Options to override defaults
    cython_args: ext_dict.get('cython_args', []),                  # Use Cython specific arguments if any
    c_args: ext_dict.get('c_args', []),                            # Additional C compilation arguments
    cpp_args: ext_dict.get('cpp_args', []),                        # Additional C++ compilation arguments
    install: ext_dict.get('install', true),                        # Install the .so file this extension module
    subdir: ext_dict.get('subdir', '.'),                           # Subdirectory where the .so file module will be placed
    # install_dir: ext_dict.get('install_dir', '.'),                 # Subdirectory where the .so file module will be installed
  )
endforeach