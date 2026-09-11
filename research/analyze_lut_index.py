"""Analyse the retained LUT index probe (see `lut_index_probe/README.md`).

Reads the board outputs written by `build_lut_index_probe.py` plus the four
sign-aware ramp windows per stem and answers three questions:

1. Is the Q15-value -> output-code conversion `clip(round(v*255/32768) - 128)`?
   Checked on the control stem, whose index `512 + 2q` is already verified.
2. Does the hardware index equal the emitted reference `512 + round(64*G*x)`
   (G = H on the negative half, 1 on the positive half)?
3. Is the measured index an affine fixed-point function of the input code,
   `floor((A*q + B) / 2**J)`?  If some (A, B, J) reproduced the measurement the
   non-power-of-two band could be emitted exactly; the search is the retained
   negative result.

Development-only: no board or vendor object is needed, the retained outputs are
enough.
"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "lut_index_probe"
K = 255.0 / 32768.0  # Q15 table value -> 8-bit output code scale (Sigmoid).
JMAX = 14


def decode_context(manifest=None):
    """Flat (code, channel) samples shared by every probe model."""
    return [(int((j * 64 + p + 85 * c) % 256) - 128, c)
            for j in range(4) for p in range(64) for c in range(3)]


def code_of(index, neg_pivot, pos_pivot, slope=128):
    """Predicted output code of the sign-aware ramp table at one index."""
    value = (neg_pivot - index) * slope if index < 512 else (index - pos_pivot) * slope
    value = min(max(value, -32768), 32767)
    return int(min(max(int(math.floor(value * K + 0.5)) - 128, -128), 127))


def measured_index(stem, manifest):
    """Intersect the four ramp windows; a unique index is an exact measurement."""
    base = stem * 10
    models = [manifest[base + k] for k in range(1, 5)]
    codes = [[b - 256 if b > 127 else b
              for b in (PROBE / f"out{base + k:03}.i8").read_bytes()]
             for k in range(1, 5)]
    samples = decode_context(manifest)
    out = []
    for s, _ in enumerate(samples):
        candidates = set(range(0, 1025))
        for model, code in zip(models, codes):
            candidates = {i for i in candidates
                          if code_of(i, model["neg_pivot"], model["pos_pivot"]) == code[s]}
            if not candidates:
                break
        out.append(next(iter(candidates)) if len(candidates) == 1 else None)
    return out


def affine_rule_exists(pairs, jmax=JMAX):
    """Is there an integer (A, B, J) with floor((A*q + B)/2**J) == m for every pair?"""
    if not pairs:
        return []
    hits = []
    for j in range(0, jmax + 1):
        step = 1 << j
        q0, m0 = pairs[0]
        centre = int(round(m0 * step / q0)) if q0 else 0
        for a in range(centre - 4 * step - 8, centre + 4 * step + 9):
            lo = hi = None
            for q, m in pairs:
                low, high = m * step - a * q, (m + 1) * step - 1 - a * q
                lo = low if lo is None else max(lo, low)
                hi = high if hi is None else min(hi, high)
                if lo > hi:
                    break
            if lo <= hi:
                hits.append((j, a, lo, hi))
    return hits


def main():
    manifest = {m["index"]: m for m in
                json.loads((PROBE / "manifest.json").read_text())}
    samples = decode_context(manifest)
    summary = {}
    for stem in sorted({index // 10 for index in manifest}):
        base = stem * 10
        meta = manifest[base]
        gain, hardware_gain = meta["gain"], meta["hardware_gain"]
        indices = measured_index(stem, manifest)
        readable = [i for i in indices if i is not None]
        predicted = []
        pos_pairs, neg_pairs = {}, {}
        for s, (q, _) in enumerate(samples):
            g = hardware_gain if q < 0 else 1.0
            offset = int(round(64 * gain * q * g))
            predicted.append(512 + offset)
            if indices[s] is not None:
                pairs = neg_pairs if q < 0 else pos_pairs
                pairs.setdefault(q, []).append(indices[s] - 512)
        diffs = [indices[s] - predicted[s] for s in range(len(samples))
                 if indices[s] is not None]
        # True-table byte comparison against the emitted reference.
        board = (PROBE / f"out{base:03}.i8").read_bytes()
        reference = (PROBE / f"expected{base:03}.i8").read_bytes()
        board = [b - 256 if b > 127 else b for b in board]
        reference = [b - 256 if b > 127 else b for b in reference]
        mismatches = sum(1 for a, b in zip(board, reference) if a != b)
        worst = max(abs(a - b) for a, b in zip(board, reference))
        pos = [(q, max(set(v), key=v.count)) for q, v in sorted(pos_pairs.items())]
        neg = [(q, max(set(v), key=v.count)) for q, v in sorted(neg_pairs.items())]
        pos_hits, neg_hits = affine_rule_exists(pos), affine_rule_exists(neg)
        summary[meta["name"]] = dict(
            ratio=meta["ratio"], hardware_gain=hardware_gain,
            readable=len(readable), index_exact=sum(1 for d in diffs if d == 0),
            index_signed={str(v): diffs.count(v) for v in sorted(set(diffs))},
            bytes_mismatch=mismatches, bytes_worst=worst,
            pos_pairs=len(pos), neg_pairs=len(neg),
            pos_affine=len(pos_hits), neg_affine=len(neg_hits))
        print(f"{meta['name']:11s} ratio {meta['ratio']:7.4f} H {hardware_gain:.0f} "
              f"index readable {len(readable):3d} exact {summary[meta['name']]['index_exact']:3d} "
              f"bytes mismatch {mismatches:3d} worst {worst} "
              f"affine pos/neg {len(pos_hits)}/{len(neg_hits)}")
    (PROBE / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", PROBE / "analysis.json")


if __name__ == "__main__":
    main()
