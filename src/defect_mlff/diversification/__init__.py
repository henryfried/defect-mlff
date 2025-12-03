import logging
from typeguard import typechecked

__all__ = ["typechecked"]
# only runs once, the first time anything in my_package is imported
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S"
)