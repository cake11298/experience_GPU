"""pytest configuration: add repo root to sys.path so 'src' and 'experiments' are importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
