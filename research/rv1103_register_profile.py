"""Research import compatibility."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_rknpu.register_profile import REGISTERS
