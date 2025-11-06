import os
from pathlib import Path

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURE_DATA_DIR = os.path.join(DATA_DIR, "features")
