"""Research import compatibility; implementation lives in the open package."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_rknpu.quantization import Quantization, quantize, reference, receptive_fields
