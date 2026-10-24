"""Entry point: python -m mission_im_dancable"""

from __future__ import annotations

import sys

from .app import run
from .config import parse_args

if __name__ == "__main__":
    sys.exit(run(parse_args()))
