## scikit-plots/scikitplot/meson.build

######################################################################
## Install actual python sources (.py)
## https://mesonbuild.com/Python-module.html#install_sources
######################################################################

# Not Needed we also copied before all files via "install_subdir"
# py_sources = [
#   '__init__.py',
# ]
# py.install_sources(
#   py_sources,
#   subdir: 'scikitplot'
#   install_tag: 'python-runtime', # default
# )

######################################################################
## Templated file: '__config__.py.in'
## https://mesonbuild.com/Reference-manual_functions.html#configure_file
######################################################################

conf_data = configuration_data()
conf_data.set('CROSS_COMPILED', meson.is_cross_build())

# Python information
conf_data.set('PYTHON_PATH', py.full_path())
conf_data.set('PYTHON_VERSION', py.language_version())

machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}
# Machines CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

# Define the compilers dictionary
compilers = {
  'C': cc,
  'CPP': cpp,
  'CYTHON': cy,
}
# Conditionally add Fortran compiler if enabled
if get_option('fortran').enabled()
  compilers += {'FORTRAN': ff}
endif
# Add `pythran` information if present
if get_option('pythran').enabled()
  conf_data.set(
    'PYTHRAN_VERSION', pythran.version(), description: 'pythran version')
  conf_data.set(
    'PYTHRAN_INCDIR', incdir_pythran, description: 'pythran path')
endif
# Set compiler information
foreach name, compiler : compilers
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
  conf_data.set(name + '_COMP_ARGS', ', '.join(
      get_option(name.to_lower() + '_args')
    )
  )
  conf_data.set(name + '_COMP_LINK_ARGS', ', '.join(
      get_option(name.to_lower() + '_link_args')
    )
  )
endforeach

# Dependencies information, If Needed
dependency_map = {
  'PYBIND11': dep_pybind11,
}

# Dependencies information
foreach name, dep : dependency_map
  conf_data.set(name + '_NAME', dep.name())
  conf_data.set(name + '_FOUND', dep.found())
  if dep.found()
    conf_data.set(name + '_VERSION', dep.version())
    conf_data.set(name + '_TYPE_NAME', dep.type_name())
    conf_data.set(name + '_INCLUDEDIR', dep.get_variable('includedir', default_value: 'unknown'))
    conf_data.set(name + '_LIBDIR', dep.get_variable('libdir', default_value: 'unknown'))
    conf_data.set(name + '_OPENBLAS_CONFIG', dep.get_variable('openblas_config', default_value: 'unknown'))
    conf_data.set(name + '_PCFILEDIR', dep.get_variable('pcfiledir', default_value: 'unknown'))
  endif
endforeach

configure_file(
  input: '__config__.py.in',
  output: '__config__.py',
  configuration: conf_data,
  install_dir: py.get_install_dir() / 'scikitplot',
  install: true,
  install_tag: 'python-runtime',
)

######################################################################
## Generate 'version.py' for "--sdist"
## "--sdist" require '.git' if error use only "--wheel" packaging
######################################################################

# Generate version.py for sdist
meson.add_dist_script(
  '_build_utils/gitversion.py',       # The script to execute
  '--meson-dist',                      # Additional arguments
  '--write', 'scikitplot/version.py',  # Additional arguments
)
if not fs.exists('version.py')
  generate_version = custom_target('generate-version',
    build_always_stale: true,
    build_by_default: true,
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    input: '_build_utils/gitversion.py',
    install: true,
    install_dir: py.get_install_dir() / 'scikitplot',
    install_tag: 'python-runtime',
    output: 'version.py',
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['version.py'],
    subdir: 'scikitplot',
    install_tag: 'python-runtime', # default
  )
endif

######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# Need for Cython cimports across subpackages to work, i.e. avoid errors like
# relative cimport from non-package directory is not allowed
_cython_tree = [
  fs.copyfile('__init__.py'),
  fs.copyfile('_numcpp_api.pxd'),
  # Pure Python types.
  fs.copyfile('__init__.pyi'),
  fs.copyfile('_c_internal_utils.pyi'),
  fs.copyfile('_docstring.pyi'),
  fs.copyfile('_preprocess.pyi'),
  fs.copyfile('cbook.pyi'),
  fs.copyfile('version.pyi'),
]

######################################################################
## Next
######################################################################

# Subpackages are mostly in alphabetical order except to handle Cython
# dependencies across subpackages
# Ordering of subdirs: because other submodules have dependencies on cython.
# After those, subdirs with the most heavy builds should come first
# (that parallelizes better)
# belows can be needs to be to be before ... since ... cimport *.pxd
subdir('_xp_core_lib')
subdir('_numcpp_api')
subdir('api')
subdir('stats')