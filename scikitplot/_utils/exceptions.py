# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains errors/exceptions and warnings of general use for
scikitplot. Exceptions that are specific to a given subpackage should *not* be
here, but rather in the particular subpackage.
"""

__all__ = [
    "ScikitplotWarning",
    "ScikitplotUserWarning",
    "ScikitplotDeprecationWarning",
    "ScikitplotPendingDeprecationWarning",
    "ScikitplotBackwardsIncompatibleChangeWarning",
    "DuplicateRepresentationWarning",
    # "NoValue",  # see: globals
]


class ScikitplotWarning(Warning):
    """
    The base warning class from which all Scikitplot warnings should inherit.

    Any warning inheriting from this class is handled by the Scikitplot logger.
    """


class ScikitplotUserWarning(UserWarning, ScikitplotWarning):
    """
    The primary warning class for Scikitplot.

    Use this if you do not need a specific sub-class.
    """


class ScikitplotDeprecationWarning(ScikitplotWarning):
    """
    A warning class to indicate a deprecated feature.
    """


class ScikitplotPendingDeprecationWarning(PendingDeprecationWarning, ScikitplotWarning):
    """
    A warning class to indicate a soon-to-be deprecated feature.
    """


class ScikitplotBackwardsIncompatibleChangeWarning(ScikitplotWarning):
    """
    A warning class indicating a change in astropy that is incompatible
    with previous versions.

    The suggested procedure is to issue this warning for the version in
    which the change occurs, and remove it for all following versions.
    """


class DuplicateRepresentationWarning(ScikitplotWarning):
    """
    A warning class indicating a representation name was already registered.
    """


# class _NoValue:
#     """Special keyword value.

#     This class may be used as the default value assigned to a
#     deprecated keyword in order to check if it has been given a user
#     defined value.
#     """

#     def __repr__(self):
#         return "astropy.utils.exceptions.NoValue"


# NoValue = _NoValue()


def __getattr__(name: str):
    if name in ("ErfaError", "ErfaWarning"):
        import warnings

        warnings.warn(
            f"Importing {name} from astropy.utils.exceptions was deprecated "
            "in version 6.1 and will stop working in a future version. "
            f"Instead, please use\nfrom erfa import {name}\n\n",
            category=ScikitplotDeprecationWarning,
            stacklevel=1,
        )

        import erfa

        return getattr(erfa, name)

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")