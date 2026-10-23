"""Make `python3 -m pytest python/tests -q` work from the repo root (tests import gai)."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
