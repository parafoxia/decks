__all__ = ("DATA_DIR", "cli", "DECKS", "DecksNet", "build_net", "train")

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

from . import cli
from .context import DECKS
from .data import *
from .model import DecksNet
from .nets import build_net
from .training import train
