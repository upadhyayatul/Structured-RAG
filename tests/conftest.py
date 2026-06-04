"""
Pytest configuration file.
Sets up the necessary Python path for importing the main package
during test execution and provides shared test fixtures.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
