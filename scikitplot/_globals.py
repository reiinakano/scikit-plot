"""This module was copied from the numpy project.

Module defining global singleton classes.

This module raises a RuntimeError if an attempt to reload it is made. In that
way the identities of the classes defined here are fixed and will remain so
even if scikitplot itself is reloaded. In particular, a function like the following
will still work correctly after scikitplot is reloaded::

    def foo(arg=np._NoValue):
        if arg is np._NoValue:
            ...

That was not the case when the singleton classes were defined in the scikitplot
``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
motivated this module.

"""
from __future__ import division, absolute_import, print_function

__ALL__ = [
  'ModuleDeprecationWarning',
  'VisibleDeprecationWarning',
  '_Default',
  '_NoValue',
  '_Deprecated',
]

######################################################################
## Module Disallow reloading
######################################################################

# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading scikitplot._globals is not allowed')
_is_loaded = True

######################################################################
## ModuleDeprecationWarning
######################################################################

class ModuleDeprecationWarning(DeprecationWarning):
    """Module deprecation warning.

    The nose tester turns ordinary Deprecation warnings into test failures.
    That makes it hard to deprecate whole modules, because they get
    imported by default. So this is a special Deprecation warning that the
    nose tester will let pass without making tests fail.

    """


ModuleDeprecationWarning.__module__ = 'scikitplot'

######################################################################
## VisibleDeprecationWarning
######################################################################

class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.

    """


VisibleDeprecationWarning.__module__ = 'scikitplot'

######################################################################
## Singleton variables
######################################################################

class SingletonBase:
    """
    A base class for singleton objects.

    Ensures that only one instance of a class exists at any time. This pattern 
    is useful for classes like markers (e.g., _NoValueType, Default, Deprecated) 
    where unique instances simplify comparison and state management.

    Methods
    -------
    __new__(cls)
        Ensures only one instance of the class is created.
    __reduce__()
        Ensures that the instance is preserved during pickling (if needed).
    """

    # Class attribute to hold the single instance of the class.
    _instance = None

    def __new__(cls):
        """
        Override the default object creation method to ensure only 
        one instance of the class is created (singleton behavior).
        """
        if cls._instance is None:
            # If the instance doesn't exist, create it using the parent class's __new__.
            cls._instance = super().__new__(cls)
        return cls._instance  # Return the single instance.

    def __reduce__(self):
        """
        Ensures that the singleton instance is correctly restored 
        when the object is serialized and deserialized (pickled).
        """
        return (self.__class__, ())

class _DefaultType(SingletonBase):
    """
    A marker representing the use of a default value.

    This class is used to indicate that a parameter is set to its default value. 
    It helps to distinguish between cases where the user intentionally provided 
    a value and cases where the default behavior is used.

    Examples
    --------
    >>> default = Default()
    >>> print(default)
    <default>

    >>> another_default = Default()
    >>> default is another_default  # Singleton behavior ensures one instance
    True
    """

    def __repr__(self):
        """
        Returns a string representation of the object.
        Printing <default> makes it easy to spot when the default value is used.
        """
        return "<default>"

class _NoValueType(SingletonBase):
    """
    A special value indicating no user-defined input.

    This class provides a unique marker to detect whether a user has provided 
    a value to a function or if a default behavior should be applied.

    Examples
    --------
    >>> no_value = _NoValueType()
    >>> print(no_value)
    <no value>

    >>> another_instance = _NoValueType()
    >>> no_value is another_instance  # All instances are the same
    True
    """

    def __repr__(self):
        """
        Returns a string representation of the object.
        This helps in debugging and makes it clear when <no value> is printed.
        """
        return "<no value>"

class _DeprecatedType(SingletonBase):
    """
    A marker indicating that a value or feature is deprecated.

    This class is useful to signal that a parameter or feature is no longer recommended 
    for use and may be removed in future versions of code or APIs.

    Examples
    --------
    >>> deprecated = Deprecated()
    >>> print(deprecated)
    <deprecated>

    >>> another_deprecated = Deprecated()
    >>> deprecated is another_deprecated  # Ensures singleton behavior
    True
    """

    def __repr__(self):
        """
        Returns a string representation of the object.
        Printing <deprecated> makes it easy to identify deprecated values or features.
        """
        return "<deprecated>"

_Default = _DefaultType()
_NoValue = _NoValueType()
_Deprecated = _DeprecatedType()

######################################################################
## 
######################################################################