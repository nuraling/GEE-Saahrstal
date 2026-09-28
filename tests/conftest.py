import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("STOCKPILE_ENABLE_DEPTH", "0")   # no model download in unit tests
