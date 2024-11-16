"""
The typing module: Support for gradual typing as defined by PEP 484 and subsequent PEPs.

This module contains Type aliases which are useful for scikitplot and potentially
downstream libraries.

.. admonition:: Provisional status of typing

    The ``typing`` module and type stub files are considered provisional and may change
    at any time without a deprecation period.

Among other things, the module includes the following:
* Generic, Protocol, and internal machinery to support generic aliases.
  All subscripted types like X[int], Union[int, str] are generic aliases.
* Various "special forms" that have unique meanings in type annotations:
  NoReturn, Never, ClassVar, Self, Concatenate, Unpack, and others.
* Classes whose instances can be type arguments to generic classes and functions:
  TypeVar, ParamSpec, TypeVarTuple.
* Public helper functions: get_type_hints, overload, cast, final, and others.
* Several protocols to support duck-typing:
  SupportsFloat, SupportsIndex, SupportsAbs, and others.
* Special types: NewType, NamedTuple, TypedDict.
* Deprecated aliases for builtin types and collections.abc ABCs.

Any name not present in __all__ is an implementation detail
that may be changed without notice. Use at your own risk!
"""
from __future__ import annotations

import abc
import numbers
import numpy as np

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Hashable, Sequence
from datetime import date, datetime, timedelta
from fractions import Fraction

from enum import Enum, EnumType
from types import GenericAlias, NoneType, ModuleType
from types import DynamicClassAttribute, LambdaType

from typing import TYPE_CHECKING
from typing import TypeVar, TypeAlias, Generic, Any
from typing import Callable, Literal, Optional, Union
from typing import List, Tuple, Dict, NamedTuple, TypedDict

import numpy.typing as np_typing
from numpy.typing import NDArray
from numpy import ndarray  # TODO use ArrayLike?
from pandas import Series, Index, Timestamp, Timedelta
from matplotlib.colors import Colormap, Normalize

_HT = TypeVar("_HT", bound=Hashable)
DT = TypeVar("DT", bound=np.generic)

from .._globals import (
  _DefaultType as Default,
  _NoValueType,
  _DeprecatedType as Deprecated,
)

ColumnName = Union[
    str, bytes, date, datetime, timedelta, bool, complex, Timestamp, Timedelta
]
Vector = Union[Series, Index, ndarray]

VariableSpec = Union[ColumnName, Vector, None]
VariableSpecList = Union[List[VariableSpec], Index, None]

# A DataSource can be an object implementing __dataframe__, or a Mapping
# (and is optional in all contexts where it is used).
# I don't think there's an abc for "has __dataframe__", so we type as object
# but keep the (slightly odd) Union alias for better user-facing annotations.
DataSource = Union[object, Mapping, None]

OrderSpec = Union[Iterable, None]  # TODO technically str is iterable
NormSpec = Union[Tuple[Optional[float], Optional[float]], Normalize, None]

# TODO for discrete mappings, it would be ideal to use a parameterized type
# as the dict values / list entries should be of specific type(s) for each method
PaletteSpec = Union[str, list, dict, Colormap, None]
DiscreteValueSpec = Union[dict, list, None]
ContinuousValueSpec = Union[
    Tuple[float, float], List[float], Dict[Any, float], None,
]


# class Default(Enum):
#     def __repr__(self):
#         return "<default>"

# class Deprecated(Enum):
#     def __repr__(self):
#         return "<deprecated>"

default = Default()
deprecated = Deprecated()