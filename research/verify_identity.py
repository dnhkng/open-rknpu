"""Check all logical outputs of the deterministic fixture against integer math."""
from pathlib import Path
import sys
import json
import re

path = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent / "identity_board_c.log"
cases = {}
permutation = [0, 1, 2]
height, width = 8, 8
if len(sys.argv) > 2:
    reference = json.loads(Path(sys.argv[2]).read_text())
    permutation = reference["input_channel_for_output"]
    _, height, width, _ = reference["shape_nhwc"]
text = path.read_text()
native_h, native_w = height, width
match = re.search(r"ATTR 9 .*dims=1,1,(\d+),(\d+),16,", text)
if match:
    native_h, native_w = map(int, match.groups())
for line in text.splitlines():
    if line.startswith("OUTPUT "):
        _, name, value = line.split()
        cases[name] = bytes.fromhex(value)
assert set(cases) == {"zero", "constant", "ramp", "impulse"}, "missing cases"
for name, data in cases.items():
    assert len(data) == native_h*native_w*16, "unexpected native output size"
    for h in range(height):
        for w in range(width):
            for c in range(3):
                source = permutation[c]
                value = 128 if name == "constant" else (h*width*3+w*3+source)%256 if name == "ramp" else 255 if name == "impulse" and (h,w,source)==(3,4,1) else 0
                # This fixture has output scale 0.75, zero point -128;
                # channel multipliers are 0.25, 0.5, 0.75. No half ties occur.
                expected = ((value*(c+1)+1)//3 - 128) & 255
                actual = data[(h*native_w+w)*16+c]
                assert actual == expected, (name, h, w, c, actual, expected)
    print(name + ": %d/%d logical INT8 outputs match exactly" % (height*width*3,height*width*3))
print("PASS: %d values; padding bytes intentionally excluded" % (height*width*3*4))
