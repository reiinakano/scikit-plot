"""
Updated:  August 17 2024
Author :  Muhammed Çelik

Setup script for installing scikit-plots

For license information, see LICENSE and/or NOTICE.md
"""
from __future__ import (
    print_function
)
import os
import io
import sys
import codecs
import pathlib

from setuptools import (
    setup,
    find_packages,
    find_namespace_packages,
    Extension,
)

##########################################################################
## Package Information
##########################################################################

## Basic information
## Get the directory where setup.py is located
HERE         = pathlib.Path(__file__).parent
NAME         = 'scikit-plots'
VERSION_PATH = os.path.join(HERE, 'scikitplot/__init__.py')
DESCRIPTION  = 'An intuitive library to add plotting functionality to scikit-learn objects.'
## Read the contents of the README file
README       = 'README.md'
PKG_DESCRIBE = (HERE / README).read_text(encoding='utf-8')
## Define the keywords
KEYWORDS = [
    "matplotlib",
    "scikit-learn",
    "xgboost",
    "catboost",
    "tensorflow",
    "keras",
    "pytorch",
    "machine learning",
    "data science",
    "visualization",
]
# LICENSE      = 'MIT License'  # deprecated
LICENSE_FILES= 'LICEN[CS]E*'
## If your name first as you're the current maintainer
AUTHOR       = 'Reiichiro Nakano et al.'
A_EMAIL      = 'reiichiro.s.nakano@gmail.com'
MAINTAINER   = 'Muhammed Çelik'
M_EMAIL      = 'muhammed.business.network@gmail.com'
REQUIRE_PATH = 'requirements.txt'

# Project homepage, often a link to GitHub or GitLab
# Often specified in the [project] table
HOMEPAGE      = "https://scikit-plots.github.io"  # alias URL
DOWNLOAD_URL  = "https://github.com/celik-muhammed/scikit-plot/tree/muhammed-dev"
DOCUMENTATION = "https://scikit-plots.github.io/stable/"
DONATE        = "https://github.com/celik-muhammed/scikit-plot#donate"
FORUM         = "https://github.com/celik-muhammed/scikit-plot/issues"
ISSUES        = "https://github.com/celik-muhammed/scikit-plot/issues"
REPO_FORKED   = "https://github.com/reiinakano/scikit-plot"
REPOSITORY    = "https://github.com/scikit-plots/scikit-plots"
# The changelog, really useful for ongoing users of your project
CHANGELOG     = "https://scikit-plots.github.io/dev/whats_new/whats_new.html"

## Directories to ignore in find_packages
EXCLUDES = [
    ".binder", ".binder.*",
    "auto_building_tools",
    "docs", "docs.*",
    "galleries", "galleries.*",
    "examples", "examples.*",
    "notebooks", "notebooks.*",
    "tests", "tests.*",
    "third_party", "third_party.*",
    "paper", "paper.*",
    "register",
    "fixtures",
    "bin",
]
PACKAGE = find_packages(
    where='.', 
    include=['scikitplot', 'scikitplot.*'], 
    exclude=EXCLUDES
)

## experimental c and c++
## from Cython.Build import cythonize
## import numpy as np
# extension_mod = Extension(
#     # 'mypackage.mymodule'
#     name                = '*',
#     # Only include .cpp, .cxx, .cc, .c, etc. compiler find .h files
#     # Cython compilation include compile .pyx, to find (.pxd, pxi) files
#     sources             = [
#         'scikitplot/experimental/_faddeeva.cxx',                   # C++ Source File
#         'scikitplot/experimental/_special_ufuncs_docs.cpp',        # C++ Source File
#         'scikitplot/experimental/_special_ufuncs.cpp',             # C++ Source File
#         'scikitplot/experimental/cython_special.pyx',              # C++ Source File
#         'scikitplot/experimental/Faddeeva.cc',                     # C++ Source File
#         'scikitplot/experimental/sf_error_state.c',                # C Source File
#         'scikitplot/experimental/sf_error.cc',                     # C++ Source File
#         'scikitplot/experimental/xsf_wrappers.cpp',                # C++ Source File
#     ],                                                             # list of Source Files (to compile)
#     # Include NumPy header files
#     # list of directories to search for C/C++ header files (in Unix form for portability)
#     include_dirs        = [np.get_include(), 'scikitplot/experimental'],
#     py_limited_api      = True,                                    # opt-in flag for the usage of Python's limited API <python:c-api/stable>.
#     extra_compile_args=['-std=c++17'],                             # libraries and codebases expect C++11 or newer standards    
#     # extra_objects=['path/to/extra/object/files']                 # Specify the output directory if necessary
#     # language='c++',                                              # Specifies that the extension uses C++
# )

##########################################################################
## Helper Functions
##########################################################################


def read(*filenames, **kwargs):
    """
    Assume UTF-8 encoding and return the contents of the file located at the
    absolute path from the REPOSITORY joined with *filenames.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(os.path.join(here, filename), encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


def get_version(rel_path=VERSION_PATH):
    """
    Reads the python file defined in the VERSION_PATH to find the get_version
    function.
    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError('Unable to find version string.')


def get_description_type(path=README):
    """
    Returns the long_description_content_type based on the extension of the
    package describe path (e.g. .txt, .rst, or .md).
    """
    ext = pathlib.Path(path).suffix
    return {'.rst': 'text/x-rst', '.txt': 'text/plain', '.md': 'text/markdown'}[ext]


def get_requires(path=REQUIRE_PATH):
    """
    Yields a generator of requirements as defined by the REQUIRE_PATH which
    should point to a requirements.txt output by `pip freeze`.
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line

##########################################################################
## Define the configuration
##########################################################################

## https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#metadata
## https://setuptools.pypa.io/en/latest/deprecated/distutils/apiref.html#distutils.core.setup
config = {
    # packages': find_packages(),  # Finds all packages automatically
    'packages': PACKAGE,
    'include_package_data': True,
    'name': NAME,
    'version': get_version(),
    'description': DESCRIPTION,
    'long_description': PKG_DESCRIBE,
    'long_description_content_type': get_description_type(),
    'keywords': KEYWORDS,
    # 'license': LICENSE,
    'license_files': LICENSE_FILES,
    'author': AUTHOR,
    'author_email': A_EMAIL,
    'maintainer': MAINTAINER,
    'maintainer_email': M_EMAIL,
    'url': HOMEPAGE,
    'download_url': DOWNLOAD_URL,
    'project_urls': {
        'Homepage   '  : HOMEPAGE,
        'Download'     : DOWNLOAD_URL,
        'Documentation': DOCUMENTATION,
        'Donate'       : DONATE,
        'Forum'        : FORUM,
        'Issues'       : ISSUES,
        'Repo_Forked'  : REPO_FORKED,
        'Repository'   : REPOSITORY,
        'changelog'    : CHANGELOG,
    },
    'classifiers': [
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # 'entry_points': {"console_scripts": []},
    'python_requires': '>=3',
    'setup_requires': [
        'setuptools>=42',
        'wheel',   # Required to build wheels
        'pytest',
        "cython",  # Required for compiling Cython extensions
    ],  # other build-time dependencies
    # pip install -e .
    'install_requires': list(get_requires()),  # other runtime dependencies
    # pip install -e .[dev]
    'extras_require': {
        'testing': ['pytest'],
    },
    'platforms': 'any',
    # from auto_building_tools.yanked.test_commands import PyTest
    # 'cmdclass': {'test': PyTest},
    'cmdclass': {'test': 'pytest'},
    # 'ext_modules': [extension_mod],
    # 'ext_modules': cythonize(extension_mod),
}

##########################################################################
## Run setup script
##########################################################################

if __name__ == "__main__":
    setup(**config)