import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile

# cbook must import matplotlib only within function
# definitions, so it is safe to import from it here.
from . import _api, cbook, _docstring

_log = logging.getLogger(__name__)

######################################################################
## matplotlib like wrapper
######################################################################

def _replacer(data, value):
    """
    Either returns ``data[value]`` or passes ``data`` back, converts either to
    a sequence.
    """
    try:
        # if key isn't a string don't bother
        if isinstance(value, str):
            # try to use __getitem__
            value = data[value]
    except Exception:
        # key does not exist, silently fall back to key
        pass
    return cbook.sanitize_sequence(value)


def _label_from_arg(y, default_name):
    try:
        return y.name
    except AttributeError:
        if isinstance(default_name, str):
            return default_name
    return None


def _add_data_doc(docstring, replace_names):
    """
    Add documentation for a *data* field to the given docstring.

    Parameters
    ----------
    docstring : str
        The input docstring.
    replace_names : list of str or None
        The list of parameter names which arguments should be replaced by
        ``data[name]`` (if ``data[name]`` does not throw an exception).  If
        None, replacement is attempted for all arguments.

    Returns
    -------
    str
        The augmented docstring.
    """
    if (docstring is None
            or replace_names is not None and len(replace_names) == 0):
        return docstring
    docstring = inspect.cleandoc(docstring)

    data_doc = ("""\
    If given, all parameters also accept a string ``s``, which is
    interpreted as ``data[s]`` if ``s`` is a key in ``data``."""
                if replace_names is None else f"""\
    If given, the following parameters also accept a string ``s``, which is
    interpreted as ``data[s]`` if ``s`` is a key in ``data``:

    {', '.join(map('*{}*'.format, replace_names))}""")
    # using string replacement instead of formatting has the advantages
    # 1) simpler indent handling
    # 2) prevent problems with formatting characters '{', '%' in the docstring
    if _log.level <= logging.DEBUG:
        # test_data_parameter_replacement() tests against these log messages
        # make sure to keep message and test in sync
        if "data : indexable object, optional" not in docstring:
            _log.debug("data parameter docstring error: no data parameter")
        if 'DATA_PARAMETER_PLACEHOLDER' not in docstring:
            _log.debug("data parameter docstring error: missing placeholder")
    return docstring.replace('    DATA_PARAMETER_PLACEHOLDER', data_doc)


# The decorator modifies how arguments are passed to `func`
# and how data can be handled when provided as a `data` argument.
# The decorator attempts to replace arguments from the data dictionary
# for any argument in the function signature, if User-provided Not replaced args.
def _preprocess_data(func=None, *, replace_names=None, label_namer=None):
    """
    A decorator to add a 'data' kwarg to a function.

    When applied::

        @_preprocess_data()
        def func(*args, **kwargs): ...

    the signature is modified to ``decorated(*args, data=None, **kwargs)``
    with the following behavior:

    - if called with ``data=None``, forward the other arguments to ``func``;
    - otherwise, *data* must be a mapping; for any argument passed in as a
      string ``name``, replace the argument by ``data[name]`` (if this does not
      throw an exception), then forward the arguments to ``func``.

    In either case, any argument that is a `MappingView` is also converted to a
    list.

    Parameters
    ----------
    replace_names : list of str or None, default: None
        The list of parameter names for which lookup into *data* should be
        attempted. If None, replacement is attempted for all arguments.
    label_namer : str, default: None
        If set e.g. to "namer" (which must be a kwarg in the function's
        signature -- not as ``**kwargs``), if the *namer* argument passed in is
        a (string) key of *data* and no *label* kwarg is passed, then use the
        (string) value of the *namer* as *label*. ::

            @_preprocess_data(label_namer="foo")
            def func(foo, label=None): ...

            func("key", data={"key": value})
            # is equivalent to
            func.__wrapped__(value, label="key")
    """

    if func is None:  # Return the actual decorator.
        return functools.partial(
            _preprocess_data,
            replace_names=replace_names, label_namer=label_namer)

    sig = inspect.signature(func)
    varargs_name = None
    varkwargs_name = None
    arg_names = []
    params = list(sig.parameters.values())
    for p in params:
        if p.kind is Parameter.VAR_POSITIONAL:
            varargs_name = p.name
        elif p.kind is Parameter.VAR_KEYWORD:
            varkwargs_name = p.name
        else:
            arg_names.append(p.name)
    data_param = Parameter("data", Parameter.KEYWORD_ONLY, default=None)
    if varkwargs_name:
        params.insert(-1, data_param)
    else:
        params.append(data_param)
    new_sig = sig.replace(parameters=params)
    arg_names = arg_names[:]  # [1:] remove the first "ax" / self arg

    assert {*arg_names}.issuperset(replace_names or []) or varkwargs_name, (
        "Matplotlib internal error: invalid replace_names "
        f"({replace_names!r}) for {func.__name__!r}")
    assert label_namer is None or label_namer in arg_names, (
        "Matplotlib internal error: invalid label_namer "
        f"({label_namer!r}) for {func.__name__!r}")

    @functools.wraps(func)
    def inner(*args, data=None, **kwargs):
        if data is None:
            return func(
                # ax,
                *map(cbook.sanitize_sequence, args),
                **{k: cbook.sanitize_sequence(v) for k, v in kwargs.items()})

        bound = new_sig.bind(
            # ax,
            *args,
            **kwargs,
        )
        auto_label = (
            bound.arguments.get(label_namer) or bound.kwargs.get(label_namer)
        )

        for k, v in bound.arguments.items():
            if k == varkwargs_name:
                for k1, v1 in v.items():
                    if replace_names is None or k1 in replace_names:
                        v[k1] = _replacer(data, v1)
            elif k == varargs_name:
                if replace_names is None:
                    bound.arguments[k] = tuple(_replacer(data, v1) for v1 in v)
            else:
                if replace_names is None or k in replace_names:
                    bound.arguments[k] = _replacer(data, v)

        new_args = bound.args
        new_kwargs = bound.kwargs

        args_and_kwargs = {**bound.arguments, **bound.kwargs}
        if label_namer and "label" not in args_and_kwargs:
            new_kwargs["label"] = _label_from_arg(
                args_and_kwargs.get(label_namer), auto_label)

        return func(*new_args, **new_kwargs)

    inner.__doc__ = _add_data_doc(inner.__doc__, replace_names)
    inner.__signature__ = new_sig
    return inner