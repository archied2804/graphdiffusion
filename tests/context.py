"""Path setup for test imports — Hitchhiker's Guide pattern."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import graph_diffusion  # noqa: E402, F401
