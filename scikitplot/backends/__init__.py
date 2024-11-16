# scikitplot/backend/__init__.py

import os

__all__ = ['init_backend']

def init_backend():
    """
    Initialize the appropriate backend for the library based on the 
    SKPLT_BACKEND environment variable.

    This function checks the value of the SKPLT_BACKEND environment variable
    and imports the corresponding backend module. The available backends
    include 'cpu', 'gpu', and 'tpu'. If an unsupported backend is specified,
    an ImportError will be raised.

    Raises
    ------
    ImportError
        If an unsupported backend is specified in the SKPLT_BACKEND variable.

    Notes
    -----
    If SKPLT_BACKEND is not set, it defaults to 'cpu'.

    Examples
    --------
    To use the GPU backend, set the environment variable:
    
    >>> import os
    >>> os.environ["SKPLT_BACKEND"] = "gpu"
    >>> init_backend()
    """
    import os

    backend = os.getenv("SKPLT_BACKEND", "cpu")

    if backend == "cpu":
        from . import cpu_backend
    elif backend == "gpu":
        from . import gpu_backend
    elif backend == "tpu":
        from . import tpu_backend
    else:
        raise ImportError(f"Unsupported backend: {backend}")
