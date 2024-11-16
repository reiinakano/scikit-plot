import sys
from typing import Final, TypeAlias

from typing_extensions import LiteralString

__all__ = (
  'git_revision',
  'version',
  '__version__',
  'full_version',
  'short_version',
  'release',
)

git_revision: Final[LiteralString]

version: Final[LiteralString]
__version__: Final[LiteralString]
full_version: Final[LiteralString]
short_version: Final[LiteralString]

release: Final[bool]