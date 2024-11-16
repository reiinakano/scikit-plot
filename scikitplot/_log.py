"""This module was copied from the setuptools_scm project.

Module Dependencies:
    - Python standard library: `logging`

logging helpers, supports vendoring

This module provides a centralized logging setup for the application.
It defines loggers with parent-child relationships and demonstrates
different logging levels (CRITICAL, ERROR, WARNING, INFO, DEBUG) to manage
log output effectively.

If you don't configure logging explicitly, the default logger is created
with the following settings:

1. Log Level: The default log level is set to WARNING, which means that 
   DEBUG and INFO messages will be ignored. Only WARNING, ERROR, and 
   CRITICAL messages will be displayed.

2. Default Handler: A StreamHandler is added by default, which sends 
   log messages to `sys.stderr`.

Logging Levels:
    - DEBUG (10): Detailed information useful during development,
      typically of interest only when diagnosing problems.
    - INFO (20): Confirmation that things are working as expected.
    - WARNING (30): An indication that something unexpected happened,
      or indicative of some problem in the near future.
    - ERROR (40): Due to a more serious problem,
      the software has not been able to perform some function.
    - CRITICAL (50): A very serious error, indicating that
      the program itself may be unable to continue running.

Parent-Child Logging:
    - `log`: A parent logger for general application-wide logging.
    - `log`: A child logger under the `log` with a specific focus (e.g., "config").::

        # Child logger (e.g., configuration-related)
        log = log.getChild("config")

Usage:
    1. Import the logger in any module::
    
        from scikitplot import log

    2. Use the logger::
    
        # No configuration; use default logger and level.
        log.debug("This won't be shown")       # Ignored
        log.info("Neither will this")          # Ignored
        log.warning("This is a warning!")      # Printed to stderr
        log.error("This is an error!")         # Printed to stderr
        log.critical("Critical issue!")        # Printed to stderr

    3. Configure log level filtering as needed::
    
        log.setLevel(logging.DEBUG)
"""

from __future__ import annotations

import os
import sys
import json
import pprint
import logging
import contextlib

from typing import IO
from typing import Iterator
from typing import Mapping

######################################################################
## Formatter
######################################################################

class PrettyJSONFormatter(logging.Formatter):
    """Custom formatter to pretty-print JSON logs."""

    def __init__(self, datefmt='%Y-%m-%d %H:%M:%S') -> None:
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "file-line_numbers": f"{record.filename}:{record.lineno}",
            "logger": record.name,
            "message": record.getMessage()
        }

        # Format JSON with custom options
        json_str = json.dumps(
            log_obj,
            indent=1,                # Pretty-print with 4 spaces
            sort_keys=False,         # Sort keys alphabetically
            separators=(',', ': ')   # Maintain default spacing
        )

        # Formatting for better alignment
        return json_str

######################################################################
## Handler
######################################################################

class AlwaysStdErrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """StreamHandler that always outputs to standard error."""
  
    def __init___(self) -> None:
        super().__init__(sys.stderr)

    @property  # type: ignore [override]
    def stream(self) -> IO[str]:
        return sys.stderr

    @stream.setter
    def stream(self, value: IO[str]) -> None:
        assert value is sys.stderr

######################################################################
## Define Handler
######################################################################

def make_default_handler() -> logging.Handler:
    """Create a default logging handler, using RichHandler if available."""
    try:
        from rich.console import Console
        from rich.logging import RichHandler
        console = Console(stderr=True)          
        handler = RichHandler(console=console)      
        handler.setFormatter(PrettyJSONFormatter())
        # raise ImportError
        return handler
    except ImportError:
        # from logging.handlers import RotatingFileHandler
        
        # handler = RotatingFileHandler(
        #     "app.log", maxBytes=5000, backupCount=2  # 5KB per file, 2 backups
        # )
      
        # Fallback to AlwaysStdErrHandler if Rich is not available
        handler = AlwaysStdErrHandler()
        # handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        # handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d - %(message)s"))
        handler.setFormatter(logging.Formatter(
            fmt='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "file-line_numbers": "%(filename)s:%(lineno)d", '
                 '"logger": "%(name)s", "message": "%(message)s"}',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        return handler

_default_handler = make_default_handler()

######################################################################
## Define the top-level logger
######################################################################

# This uses the default logging configuration
# logging.basicConfig()  # Using default settings
# logging.basicConfig(
#     # level=logging.WARN,
#     # handlers=[logging.StreamHandler()],
#     format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
#     datefmt="%Y-%m-%d %H:%M:%S"
# )


# Define the top-level logger for the project.
# Global logger variable
log = logging.getLogger(__name__.rsplit(".", 1)[0])
log.propagate = False

# log.debug("Top-level logger initialized.")

######################################################################
## add Handler
######################################################################

# Configure the logger here (only once for the entire project).
log.addHandler(_default_handler)

######################################################################
## Define-Set log level
######################################################################

def _default_log_level(_env: Mapping[str, str] = os.environ) -> int:
    val: str | None = _env.get("SKPLT_DEBUG")
    return logging.WARN if val is None else logging.DEBUG

# Optional: Set the log level for specific use cases.
log.setLevel(_default_log_level())

######################################################################
## func
######################################################################

@contextlib.contextmanager
def defer_to_pytest() -> Iterator[None]:
    log.propagate = True
    old_level = log.level
    log.setLevel(logging.NOTSET)
    log.removeHandler(_default_handler)
    try:
        yield
    finally:
        log.addHandler(_default_handler)
        log.propagate = False
        log.setLevel(old_level)


@contextlib.contextmanager
def enable_debug(handler: logging.Handler = _default_handler) -> Iterator[None]:
    log.addHandler(handler)
    old_level = log.level
    log.setLevel(logging.DEBUG)
    old_handler_level = handler.level
    handler.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        log.setLevel(old_level)
        handler.setLevel(old_handler_level)
        if handler is not _default_handler:
            log.removeHandler(handler)

######################################################################
## Expose the logger
######################################################################

# Expose the logger to other modules.
def get_logger(child: str = None) -> logging.Logger:
    """
    Retrieve a child logger with the specified name.

    Parameters
    ----------
    child : str, optional
        Name of the child logger to retrieve. If None, the root logger is returned.
    
    Returns
    -------
    logging.Logger
        The logger instance corresponding to the specified child name.
    
    Examples
    --------
    Get a root logger:
    
    .. jupyter-execute::
    
        >>> from scikitplot._log import log
        >>> log.info("This is a log message from the root logger.")
    
    Get a named child logger:
    
    .. jupyter-execute::
    
        >>> from scikitplot._log import get_logger
        >>> log = get_logger(__name__)
        >>> log.info("This is a log message from my_child logger.")
    """
    # Ensure the root logger is initialized
    global log
  
    # If a child name is provided, split it to take only the part before the last dot
    if isinstance(child, str):
        child = child.rsplit(".", 1)[0]  # Only use the parent part of the dotted name
    
    # Return the child logger if a name is specified, else the root logger
    log = log.getChild(child) if child else log
    return log

######################################################################
## 
######################################################################