__all__ = ("DATA_DIR", "cli", "context", "build_net", "train")

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

from . import cli, context
from .data import *
from .nets import build_net
from .training import train
