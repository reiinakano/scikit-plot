from functools import wraps

import numpy


def seed(func):
    """Decorator to seed the RNG before any function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        numpy.random.seed(0)
        return func(*args, **kwargs)

    return wrapper
