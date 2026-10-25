"""pytest from anywhere: the prototypes import each other and REAPer's tiny test models."""
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for p in (here, here.parent / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
