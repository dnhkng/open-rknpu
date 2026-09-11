"""Compatibility entry for research raw-buffer generation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_rknpu.compiler import compile_model, main

if __name__ == "__main__":
    main()
