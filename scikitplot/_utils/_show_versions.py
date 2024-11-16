"""Utility methods to print system info for debugging

adapted from :func:`pandas.show_versions`
"""

import platform
import sys
from threadpoolctl import threadpool_info
from pprint import pprint


## Define __all__ to specify the public interface of the module,
## not required default all belove func
__all__ = [
    '_get_sys_info',
    '_get_deps_info',
    'show_versions',
]
_all_ignore = ['platform', 'sys', 'threadpool_info']


def _get_sys_info():
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    """Overview of the installed version of main dependencies

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """
    deps = [
        "pip",
        "setuptools",
        "Cython",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "joblib",
        "threadpoolctl",
    ]

    deps_info = {}

    from importlib.metadata import PackageNotFoundError, version

    # Import __version__ here
    from scikitplot import __version__

    deps_info["scikitplot"] = __version__

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions():
    """
    Print useful debugging information

    Examples
    --------
    .. jupyter-execute::

        >>> import scikitplot
        >>> scikitplot.show_versions()
    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print("\nPython dependencies:")
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))

    # Show threadpoolctl results
    threadpool_results = threadpool_info()
    if threadpool_results:
        print()
        print("threadpoolctl info:")

        for i, result in enumerate(threadpool_results):
            for key, val in result.items():
                print(f"{key:>15}: {val}")
            if i != len(threadpool_results) - 1:
                print()