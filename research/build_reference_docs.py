"""SPDX-License-Identifier: MIT

Deterministic generator for ``docs/registers.md`` and ``docs/errors.md``.

Two generated reference documents, built from the compiler source itself so they
cannot drift from it:

* ``docs/registers.md`` renders one row per entry of
  ``open_rknpu.register_profile.REGISTERS`` (the register/default/tag table a task
  writes), with the meaning recovered from the emitters, ``runtime/sequence_format.md``
  and ``docs/investigation-log.md``.  Registers whose semantics were never recovered
  say so instead of guessing.
* ``docs/errors.md`` indexes every user-facing message the compiler raises, grouped by
  module, with a short "what it means / what to do".

Run ``python research/build_reference_docs.py`` to rewrite both files, or
``python research/build_reference_docs.py --out DIR`` to write them elsewhere (the
tests use this to prove the committed files are byte-identical to a fresh run).
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "open_rknpu"
DOCS = ROOT / "docs"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from open_rknpu.register_profile import REGISTERS  # noqa: E402

UNDECODED = "undecoded profile value"

# Meaning recovered for each profile register.  ``None`` (or a string that starts
# with :data:`UNDECODED`) means the investigation never decoded the field.  The text
# names the expression the emitters write where that is the only evidence, and cites
# the RK3588/Mesa names only where ``docs/investigation-log.md`` or ``elementwise.py``
# already did.
REGISTER_MEANINGS = {
    # --- CNA / conv geometry -------------------------------------------------
    0x100C: "CNA mode/config word; the single-input-channel profile writes 0x20008000, "
            "the depthwise second task writes 7.",
    0x1010: "CNA feature grains (bits 7:4) and scan flags (bits 3:0), written as "
            "`(feature_grains<<4)|scan_flags`; the K5 and transposed paths store 0x3ff.",
    0x1014: "CNA stride/dilation word: `(dy-1)<<21 | (dx-1)<<16 | sy<<3 | sx`; the "
            "transposed emitters use `(stride-1)<<11 | (stride-1)<<8 | 9`.",
    0x101C: UNDECODED + " (the chain profile forces 0, the 1-channel profile writes 0).",
    0x1020: "CNA input geometry: `input_width<<16 | task_input_rows`.",
    0x1024: "CNA channel geometry: `(input_channels-1)<<16 | 16-lane_plane_count`.",
    0x1028: "CNA output width in pixels.",
    0x102C: "CNA output pixel count (`output_height*output_width`).",
    0x1030: "CNA weight-table bytes: `output_channels*lanes*k*k`; depthwise stores `k*k*32`.",
    0x1034: "CNA per-output weight stride: `lanes*k*k`; depthwise stores `k*k*16`.",
    0x1038: "CNA kernel and output channels: `k<<24 | k<<16 | output_channels`.",
    0x103C: "CNA weight data entries in the high half (`data_entries<<16`); depthwise "
            "stores `(width+1)//2<<16`.",
    0x1044: "CNA input width and data entries: `input_width<<16 | data_entries`; depthwise "
            "stores `width<<16 | (width+1)//2`.",
    0x1048: "CNA input surface extent term: `((input_rows*width*tiles+2047)//2048)*0x04000000`; "
            "the transposed paths store 0x1c000000.",
    0x104C: UNDECODED + " (the chain profile forces 0, the 1-channel profile writes 0xe0).",
    0x1050: UNDECODED + " (the chain profile forces 0x10001, the 1-channel profile writes 0x14000).",
    0x1054: UNDECODED + " (the chain profile forces 0x10001).",
    0x1058: UNDECODED + " (the chain profile forces 0).",
    0x105C: UNDECODED + " (the chain profile forces 0).",
    0x1060: UNDECODED + ".",
    0x1064: UNDECODED + ".",
    0x1068: "CNA kernel phase field: `((k-1-pad_left)<<8) | (k-1-pad_top)`; depthwise "
            "stores `pad*0x101`.",
    0x106C: UNDECODED + ".",
    0x1074: UNDECODED + ".",
    0x1078: UNDECODED + " (the 1-channel profile writes 0xc00f300f).",
    0x107C: "CNA input row/line stride: `input_width*min(4,(input_rows+3)//4*2)`.",
    0x1080: "CNA input surface row atoms: `((surface_rows*input_width+3)//4)*4`.",
    0x1084: "CNA input width and rows: `input_width<<16 | input_rows`.",
    0x1088: "CNA active 16-lane plane count (`lanes`).",
    0x108C: UNDECODED + " (the chain profile forces 0).",
    0x1094: UNDECODED + ".",
    0x1100: UNDECODED + ".",
    0x1104: UNDECODED + ".",
    0x1140: UNDECODED + ".",
    0x1144: UNDECODED + ".",
    0x1188: "CNA weight-table stride: `8*k*k*tiles` (native), `8*k*k` (chain) or "
            "`k*k*8` (depthwise).",
    0x118C: "CNA feature width minus one, duplicated: `(input_width-1)<<16 | (input_width-1)`.",
    # --- DPU / activation ----------------------------------------------------
    0x3010: "DPU task control word; depthwise writes 10, the K5 transposed phase path writes 0xa.",
    0x3014: "DPU output geometry: `(output_height-1)<<16 | (output_width-1)`.",
    0x3018: "DPU output 16-lane channel count minus one (`align(oc,16)-1`; depthwise writes 31).",
    0x301C: UNDECODED + " (the chain profile forces 0, depthwise writes 10).",
    0x400C: "DPU/EW output-converter config word; 0x1e4 default, 0x1e5 for the "
            "elementwise/clamp tasks and 0x1fc for depthwise.",
    0x4010: "EW output control word; the LUT setup writes 0x804000.",
    0x4014: UNDECODED + ".",
    0x4018: "Low output clamp of the appended Mul+Clip clamp task (`lo & 0xff`).",
    0x401C: "High output clamp of the Mul+Clip clamp task (`(hi+128)&0xff`); 0x7fffffff "
            "when unused.",
    0x4024: "Output plane surface stride in bytes.",
    0x4028: "Upper accumulator clamp for the fused Clip[0,6] "
            "(`round(6/(max_weight_scale*input_scale))`).",
    0x402C: UNDECODED + " (no emitter overrides the 0x7fffffff default).",
    0x4030: "Output width minus one.",
    0x4034: "Output height minus one.",
    0x4038: UNDECODED + ".",
    0x403C: "Output channel pair: `(output_channels-1)<<16 | (aligned_lanes-1)` (15 or 31).",
    0x4040: "DPU ALU/shift config; LeakyRelu and PRelu clear its 0x1c00 ALU-selector bits, "
            "the scalar-Mul path writes 0x120050.",
    0x4044: "Negated zero point of the second Mul operand, used with the 0x120050 ALU selector.",
    0x4048: "EW control bit set by the Mul path (default 1).",
    0x404C: UNDECODED + " (the Mul+Clip clamp task forces 0).",
    0x4050: "`BS_OW_CFG` (RK3588 name): bit 0 `OW_SRC`, bit 1 `OD_BYPASS`; measured on "
            "RV1103 - `OW_SRC=1` hangs and `OD_BYPASS=1` skips the per-channel block.",
    0x4054: UNDECODED + " (the dense transposed path writes 0x8e000000).",
    0x4058: "Output channel block count: `((oc-1)//16)<<16 | 3` (native) or 3/7.",
    0x405C: "Output geometry pair: `(output_height-1)<<16 | (output_width-1)`.",
    0x4060: "Activation selector: 0x13 off, 0x12 fused Relu, 0x22 LeakyRelu/PRelu.",
    0x4064: UNDECODED + ".",
    0x4068: "Activation multiplier: LeakyRelu stores `round(alpha*16384)<<16`; PRelu stores 1 "
            "and reads its per-channel slopes from 0x502c.",
    0x406C: "Activation lower clamp: 0x80000000 when the activation is off, 0 when Relu is applied.",
    0x4070: "`DPU_EW_CFG`: EW ALU selector at bit 16 (2 = Add, 0 = Max) plus the observed "
            "mode bits; names from Mesa Rocket/RK3588, re-checked on RV1103.",
    0x4074: "Negated zero point of the chained Mul operand.",
    0x4078: "EW operand scale: 0x4000 default, 0xc000 signed for Sub, 1 for the scalar-Mul path.",
    0x407C: "EW lower output clamp; the Mul-to-Relu path forces 0.",
    0x4090: UNDECODED + ".",
    0x4094: UNDECODED + ".",
    0x4098: UNDECODED + ".",
    0x409C: UNDECODED + ".",
    0x40A0: UNDECODED + ".",
    0x40C0: "Second output surface stride (the EW/DPU write surface).",
    0x40C4: UNDECODED + ".",
    0x40D8: "Lower output clamp bound for the Mul+Clip clamp task; 0x80000000 when unused.",
    0x40DC: "Upper output clamp bound for the Mul+Clip clamp task (`hi+128`); 0x7fffffff "
            "when unused.",
    0x40E0: "Second activation lower clamp, mirroring 0x406c.",
    0x40E4: "Second upper accumulator clamp for Clip[0,6], mirroring 0x4028.",
    0x40E8: "Second output lower clamp; the Mul-to-Relu path forces 0.",
    0x40EC: UNDECODED + " (no emitter overrides the 0x7fffffff default).",
    0x4100: "LUT setup table base/width word (`0x20000 + table*0x10000`).",
    0x4104: "LUT setup table entry: one register word per lookup-table value.",
    0x4108: "LUT setup control word (1 in the setup skeleton, 0x68 on the activation path).",
    0x410C: "LUT setup word 0x50500.",
    0x4110: "LUT setup word 0xffffc000.",
    0x4114: UNDECODED + ".",
    0x4118: UNDECODED + ".",
    0x411C: "LUT setup word 0x4000.",
    0x4120: UNDECODED + ".",
    0x4124: UNDECODED + ".",
    0x4128: UNDECODED + ".",
    0x412C: UNDECODED + ".",
    0x500C: "EW output width minus one.",
    0x5010: "EW output height minus one; bits 28:16 carry `ew_line_notch_addr` (RK3588 name).",
    0x5014: "EW output 16-lane channel count minus one (15 or 31).",
    0x501C: UNDECODED + " (0, 10 and 14 across the emitter paths).",
    0x5028: "Activation coefficient-table size (8 for PRelu).",
    0x502C: "Activation coefficient-table address (the PRelu per-channel slopes).",
    0x5034: "`erdma_cfg`: `data_mode` bits 31:30, `surf_mode` bit 29, `data_size` bits 3:2, "
            "`erdma_disable` bit 0; decoded from the RK3588 TRM (`data_mode=2` hangs).",
    0x5040: "`ew_surf_stride`: stride in 16-byte atoms (16 = one atom per channel, "
            "0x240 = 36 atoms per pixel).",
    0x5044: "EW control word (0x7810/0x7816/0x907809 across the emitter paths).",
    0x5048: UNDECODED + ".",
    0x504C: UNDECODED + ".",
    0x5064: UNDECODED + ".",
    0x5068: UNDECODED + " (every emitter stores 0x01010101).",
    0x506C: "`ew_surf_notch`: pixels from the end of this feature map to the end of the shape "
            "feature map; setting it stops the compact-operand timeouts.",
    0x8028: UNDECODED + " (the pool task table stores 12).",
    0x802C: UNDECODED + " (the pool task table stores 0xffffffff).",
    # --- addresses -----------------------------------------------------------
    0x1070: "CNA input base address: the arena offset of the surface the task reads.",
    0x1110: "CNA weight-table base address (arena offset).",
    0x4020: "Output base address: the arena offset the task writes.",
    0x5018: "EW primary surface address (the accumulator-side read).",
    0x5020: "EW secondary operand base address (`RDMA_BS_BASE_ADDR`); a Conv bias block also "
            "lives here.",
    0x5038: "EW secondary surface address (the ERDMA read side / second branch).",
    # --- quantization --------------------------------------------------------
    0x1184: "Input zero point declared to the CNA (`input_zero_point-128`); it also sets the "
            "value the engine pads borders with.",
    0x4080: "Output zero point, written by the Conv/EW quantization.",
    0x4084: "Output multiplier (the Q14/Q15 conversion factor).",
    0x4088: "Output conversion shift.",
    # --- tail / control ------------------------------------------------------
    0x1004: "CNA engine control word; the LUT setup raises it to 0x30.",
    0x3004: "DPU engine control word; the LUT setup raises it to 0x30.",
    0x4004: "DPU/EW engine control word; the LUT setup raises it to 0x30.",
    0x5004: "EW engine control word; the LUT setup raises it to 0x30.",
}

_ADDRESS_REGS = {0x1070, 0x1110, 0x4020, 0x5018, 0x5020, 0x5038}
_QUANT_REGS = {0x1184, 0x4080, 0x4084, 0x4088}
_CONTROL_REGS = {0x1004, 0x3004, 0x4004, 0x5004}

# Section order for docs/registers.md.
REGISTER_GROUPS = ("CNA / conv geometry", "DPU / activation", "Addresses", "Quantization",
                   "Tail / control", "Undecoded")


def register_meaning(register: int) -> str:
    """Return the recovered meaning, or the explicit undecoded marker."""
    meaning = REGISTER_MEANINGS.get(register)
    if meaning is None:
        return UNDECODED
    return meaning


def register_group(register: int) -> str:
    """Classify a profile register into one of :data:`REGISTER_GROUPS`."""
    if register_meaning(register).startswith(UNDECODED):
        return "Undecoded"
    if register in _ADDRESS_REGS:
        return "Addresses"
    if register in _QUANT_REGS:
        return "Quantization"
    if register in _CONTROL_REGS:
        return "Tail / control"
    if 0x1000 <= register < 0x2000:
        return "CNA / conv geometry"
    return "DPU / activation"


REGISTERS_PREAMBLE = """\
This is the task register reference for the open emitters, **not** a vendor ISA
specification.  It documents the values `open_rknpu.register_profile.REGISTERS` makes a
task write: which register word, the profile default, and the engine tag.  Each row is
one entry of that table, in table order (the table lists `0x1040` twice), and the
meaning column is filled only where the investigation recovered it - an undecoded
field says so rather than inventing a vendor name.

How to read a row.  A command word is a 64-bit little-endian value laid out as
`tag << 48 | value << 16 | reg`; the address column is the low 16 bits (`reg`), the
default column is the 32-bit value word, and the tag column is the top 16 bits.  The
tag distinguishes the class of write: the profile rows carry the per-engine tags
`0x0201` (CNA), `0x0801` (DPU), `0x1001` (DPU/EW), `0x2001` (EW) and `0x0401` (pool),
while the task tail ends with `0x0101` register words (link at `0x10`, control at
`0x14`), a `0x0041` zero task descriptor and the `0x0081` enable word.  Addresses are
arena offsets.  A register the emitters do not override keeps its profile default.

The meanings come from the comments in `src/open_rknpu/register_profile.py`, the
emitters' `fields` dicts, `runtime/sequence_format.md` and `docs/investigation-log.md`
(searched for the address in hex).  Where only a field expression identifies a
register, that expression is quoted; names such as `BS_OW_CFG`, `erdma_cfg` or
`ew_surf_stride` are RK3588/Mesa names the investigation re-checked on RV1103."""


def _register_row(register: int, default: int, tag: int) -> str:
    meaning = register_meaning(register).replace("|", "\\|")
    return "| `0x%04x` | `0x%04x` | `0x%08x` | %s |" % (register, tag, default, meaning)


def build_registers_doc() -> str:
    """Render ``docs/registers.md`` from ``REGISTERS`` and the meaning table."""
    by_group = {name: [] for name in REGISTER_GROUPS}
    for register, default, tag in REGISTERS:
        by_group[register_group(register)].append((register, default, tag))
    lines = ["# Task register reference", "",
             "<!-- Generated by research/build_reference_docs.py; do not edit by hand. -->", "",
             REGISTERS_PREAMBLE, ""]
    for name in REGISTER_GROUPS:
        rows = by_group[name]
        lines.append("## %s" % name)
        lines.append("")
        if not rows:
            lines.extend(["_None._", ""])
            continue
        lines.append("| Address | Tag | Default | Meaning |")
        lines.append("| --- | --- | --- | --- |")
        lines.extend(_register_row(*row) for row in rows)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# --- error index -------------------------------------------------------------

# The three exceptions the test's independent walk looks for, plus the CLI's
# ``parser.exit(...)`` support message.
TYPED_EXCEPTIONS = ("ValueError", "RuntimeError", "KeyError")


def _canonical_expression(expression):
    """A version-independent key for a message expression.

    ``ast.unparse`` chooses quote characters per Python version (3.10/3.13 render an
    f-string with double quotes where 3.12 uses single quotes), so the raw unparse output
    cannot be a stable dictionary key. Collapsing whitespace and normalising quotes to the
    single-quote form makes the curated meanings and the drift guard portable across the
    supported interpreters.
    """
    return " ".join(expression.split()).replace('"', "'")


def _leading_literal(node):
    """Return ``(text, dynamic)`` for a raised message expression.

    ``text`` is the source literal (a plain string, a ``%`` template, or the leading
    literal of a concatenation/f-string); ``None`` means the message is a pure
    expression with no literal text at all.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False
    if isinstance(node, ast.JoinedStr):
        prefix = ""
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                prefix += value.value
            else:
                break
        return (prefix or None), True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        text, _ = _leading_literal(node.left)
        return text, True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        text, _ = _leading_literal(node.left)
        return text, True
    return None, True


def _walk_raises(path):
    """Yield ``(line, exception_name, text, dynamic, expression)`` for every typed raise."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        if not isinstance(func, ast.Name) or func.id not in TYPED_EXCEPTIONS:
            continue
        argument = node.exc.args[0] if node.exc.args else None
        text, dynamic = _leading_literal(argument) if argument is not None else (None, True)
        expression = _canonical_expression(ast.unparse(argument)) if argument is not None else ""
        yield node.lineno, func.id, text, dynamic, expression


def _walk_parser_exits(path):
    """Yield ``(line, text, dynamic, expression)`` for ``parser.exit(...)`` support messages."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "exit" or len(node.args) < 2:
            continue
        text, dynamic = _leading_literal(node.args[1])
        yield node.lineno, text, dynamic, _canonical_expression(ast.unparse(node.args[1]))


def extract_compiler_messages(src=SRC):
    """Collect every user-facing message from ``src/open_rknpu/*.py``.

    Returns a list of dicts sorted by module and line: ``module``, ``lines`` (all the
    source lines that raise it, ascending), ``text`` (literal text or ``None`` when the
    message is pure dynamic), ``dynamic`` (a formatted suffix follows the literal),
    ``expression`` (the raised Python expression) and ``source`` (``raise`` or ``exit``).
    """
    collected = {}
    for path in sorted(pathlib.Path(src).glob("*.py")):
        for line, _name, text, dynamic, expression in _walk_raises(path):
            key = (path.name, text, dynamic, expression if text is None else "")
            item = collected.setdefault(key, dict(module=path.name, text=text, dynamic=dynamic,
                                                  expression=expression))
            item.setdefault("lines", []).append(line)
        for line, text, dynamic, expression in _walk_parser_exits(path):
            key = (path.name, text, dynamic, expression if text is None else "")
            item = collected.setdefault(key, dict(module=path.name, text=text, dynamic=dynamic,
                                                  expression=expression))
            item.setdefault("lines", []).append(line)
    messages = []
    for item in collected.values():
        item["lines"] = sorted(set(item["lines"]))
        messages.append(item)
    messages.sort(key=lambda entry: (entry["module"], entry["lines"][0], entry["text"] or ""))
    return messages


def error_meaning(module, message):
    """Return the curated meaning for a message, or raise when it is missing.

    ``ERROR_MEANINGS`` keys are normally the message text (shared across modules); an
    ``(module, message)`` tuple overrides one module's row when the same text needs
    different advice there.  Pure dynamic messages are keyed by their source expression.
    """
    for key in ((module, message), message):
        if key in ERROR_MEANINGS:
            return ERROR_MEANINGS[key]
    raise KeyError("no curated meaning for compiler message: %s: %r" % (module, message))


def _escape_cell(text):
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")


def _message_cell(entry):
    if entry["text"] is None:
        return "*(dynamic message)*"
    if entry["dynamic"]:
        return "`%s` **+ dynamic suffix**" % _escape_cell(entry["text"])
    return "`%s`" % _escape_cell(entry["text"])


def _location_cell(entry):
    joined = ",".join(str(line) for line in entry["lines"])
    return "`%s:%s`" % (entry["module"], joined)


def build_errors_doc() -> str:
    """Render ``docs/errors.md`` from the messages the compiler source raises."""
    messages = extract_compiler_messages()
    by_module = {}
    for entry in messages:
        by_module.setdefault(entry["module"], []).append(entry)
    dynamic_count = sum(1 for entry in messages if entry["text"] is None)
    lines = ["# Error index", "",
             "<!-- Generated by research/build_reference_docs.py; do not edit by hand. -->", "",
             ERRORS_PREAMBLE, "",
             "This index covers %d messages across %d modules; %d are pure expressions with no "
             "literal text and are marked *dynamic*." % (len(messages), len(by_module), dynamic_count),
             ""]
    for module in sorted(by_module):
        lines.append("## `%s`" % module)
        lines.append("")
        lines.append("| Message | Location | What it means / what to do |")
        lines.append("| --- | --- | --- |")
        for entry in by_module[module]:
            lookup = entry["text"] if entry["text"] is not None else "(dynamic) " + entry["expression"]
            meaning = _escape_cell(error_meaning(entry["module"], lookup))
            lines.append("| %s | %s | %s |" % (_message_cell(entry), _location_cell(entry), meaning))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


ERRORS_PREAMBLE = """\
Every user-facing message the compiler raises, grouped by the module that raises it.
This is the support front door: paste an error and find it here.  Each row gives the
message text, the `module:line` that raises it (a comma-separated list when the same
message is raised more than once) and a one-line explanation written from the raising
code, the profile's bound and `docs/primitives.md`.

Messages a variable completes are shown as their literal prefix followed by
**+ dynamic suffix**; messages built from an expression with no literal text are marked
*(dynamic message)*.  The index is a generated file - edit
`research/build_reference_docs.py` (its meaning table) and rerun it, never this file."""


# One line per message: what it means and what to do.  Keyed by message text, with a
# ``(module, text)`` override where the same wording needs module-specific advice.
ERROR_MEANINGS = {
    # --- accuracy.py ---------------------------------------------------------
    "integer and reference outputs must share a shape":
        "The integer and float references ran on different shapes; score both on the same inputs.",
    "expected [N,C] logits and [N] labels":
        "Accuracy expects rank-2 [N,C] logits and a rank-1 [N] label vector.",
    "ranges lack measured tensors: ":
        "The calibration report has no measurement for a tensor the check needs; record it first.",
    # --- activation.py -------------------------------------------------------
    "LeakyRelu profile requires one Conv followed by LeakyRelu":
        "Only the two-node Conv -> LeakyRelu graph lowers on this path.",
    "unsupported LeakyRelu parameters/connections":
        "LeakyRelu needs only an alpha in [0,1] and must follow its Conv directly.",
    "LeakyRelu positive conversion may overflow INT32; smaller-scale lowering required":
        "The positive path can exceed the INT32 accumulator; rescale the layer or split it.",
    "PRelu profile requires one Conv followed by PRelu":
        "Only the two-node Conv -> PRelu graph lowers on this path.",
    "PRelu requires constant float32 slopes in [0,1]":
        "Supply PRelu slopes as a constant float32 tensor with every value in [0,1].",
    "PRelu slope must be scalar or [C,1,1]":
        "Only scalar or per-channel [C,1,1] PRelu slopes are lowered.",
    "PRelu positive conversion may overflow INT32; smaller-scale lowering required":
        "The positive path can exceed the INT32 accumulator; rescale the layer or split it.",
    # --- calibration.py ------------------------------------------------------
    "invalid calibration scale":
        "A calibration scale is not a positive finite float; recalibrate the graph.",
    "percentile must be in (0,100]":
        "The percentile method needs a percentile greater than 0 and at most 100.",
    "calibration method must be one of %s":
        "Pick one of the listed calibration methods (minmax, percentile, ...).",
    "calibration requires exactly one graph input":
        "Calibration only supports a single external input.",
    "calibration requires a static rank-four NCHW input":
        "The graph input must have a static rank-4 NCHW shape.",
    "calibration requires at least one Conv":
        "The graph has no Conv for calibration to measure.",
    "calibration directory must contain .npy arrays":
        "Point --calibration-data at a directory of .npy sample arrays.",
    "calibration directory produced no samples":
        "No usable .npy samples were loaded; check the calibration directory.",
    "nonfinite calibration activation: ":
        "A calibration activation is NaN or infinite; fix the weights or the samples.",
    # --- chain.py ------------------------------------------------------------
    "unsupported two-layer graph or quantization parameters":
        "The graph is not the supported 8x8 C3 Conv-Relu-Conv profile, or its quantization is out of range.",
    "native INT32 bias overflow":
        "The quantized bias exceeds INT32; widen the output scale or shrink the weights.",
    "invalid native output scale":
        "The computed output scale is zero or non-finite; check the weight range.",
    "invalid calibrated native output quantization":
        "The calibrated output scale or zero point is invalid; fix the calibration report.",
    "native output scale outside nonzero multiplier range":
        "The conversion multiplier would be 0 or above 32767; adjust the output scale.",
    "two-layer weights must have rank four":
        "Conv weights must be a rank-4 [O,I,K,K] tensor.",
    "each convolution requires a constant bias matching its output channels":
        "Both Convs need a constant bias with one value per output channel.",
    "calibration and chain output override cannot be combined":
        "Choose calibration or an explicit output range, not both.",
    # --- chain_n.py ----------------------------------------------------------
    "native chain requires [Conv, Relu]*(N-1) + [Conv]":
        "A native chain is an odd-length Conv/Relu alternation ending in Conv.",
    "native chain requires alternating Conv and Relu":
        "Node order must alternate Conv, Relu, Conv, ...",
    "native chain nodes must be connected in order":
        "Each chain node must consume the previous node's output.",
    "native chain requires one input and one output":
        "The native chain graph must be single-input, single-output.",
    "native chain external tensors must be float32 [1,3,8,8]":
        "The chain boundary is fixed at float32 [1,3,8,8].",
    "native chain convolutions require constant weights and bias":
        "Every chain Conv needs constant weight and bias initializers.",
    "native chain weights and bias must be float32 rank-four":
        "Chain weights must be float32 rank-4 and the bias float32 [C].",
    "native chain supports 1x1 or padded 3x3, stride1, group1":
        "Only K1 or symmetric-pad K3, stride 1, group 1 is lowered.",
    "native chain requires three external input and output channels":
        "The native chain boundary channel count is fixed at 3.",
    "native chain hidden channels must be 3..16 with matching bias":
        "Hidden widths are 3-16 and the next layer's input must match.",
    "native chain hidden channel counts must match between layers":
        "Adjacent Conv channel counts must line up.",
    # --- cli.py --------------------------------------------------------------
    "specify output scale and zero point together":
        "Pass --output-scale and --output-zero-point together.",
    "calibration with input quantization overrides is not supported yet":
        "Calibration cannot be combined with --input-scale/--input-zero-point.",
    "calibration and output quantization overrides cannot be combined":
        "Use calibration or explicit output quantization, not both.",
    "mutable parameters require --sequence":
        "Mutable constants and weights exist only in the sequence container; add --sequence.",
    "Mul operand zero points require --sequence":
        "Per-operand Mul zero points need the sequence container; add --sequence.",
    "unsupported target %s (supported: %s)":
        "Choose a target from the supported list; the suffix names the choices.",
    "unsupported quantization %s (supported: %s)":
        "Choose a quantization from the supported list; the suffix names the choices.",
    # --- compiler.py ---------------------------------------------------------
    "input quantization overrides currently require a single Conv[/Relu]":
        "--input-scale/--input-zero-point only apply to the single-Conv profile.",
    "network output overrides require calibration instead":
        "The legacy network profile takes no output override; calibrate instead.",
    "two-layer output quantization overrides are not supported yet":
        "The two-layer profile rejects --output-*; leave the override out.",
    "one Conv with optional following Relu is supported":
        "This path accepts Conv or Conv -> Relu only.",
    "only a directly connected standard Relu may follow Conv":
        "The Relu must be a standard attribute-free node fed by the Conv.",
    "one input/output and optional constant bias required":
        "The graph needs one input, one output and (optionally) a constant bias.",
    "unexpected graph connections":
        "Node wiring does not match the supported single-Conv chain.",
    "weights must be [O,I,K,K], I=1 or 3, 1<=O<=16, K=1, 3 or 5":
        "Reshape the Conv weights into the legacy [O,I,K,K] profile.",
    "unsupported Conv attribute: ":
        "The Conv carries an attribute the legacy profile does not implement; remove it.",
    "spatial Conv requires explicit symmetric padding K//2":
        "A K>1 Conv needs pads equal to [K//2]*4.",
    "only static NCHW [1,I,H,W], I=1 or 3, 5 <= H,W <= 8":
        "Use a static NCHW input with 1 or 3 channels and 5-8 rows/columns.",
    "input/output must be float32 tensors with matching spatial shapes":
        "Input and output must be float32 and share H/W.",
    "float32 weights required":
        "Export the Conv weights as float32.",
    "bias must be float32 [output_channels]":
        "Export one float32 bias value per output channel.",
    # --- compose.py ----------------------------------------------------------
    "compose requires at least one stage":
        "The composer needs at least one task stage.",
    "duplicate tensor name in the tensor table":
        "Two tensor descriptors share a name; make them unique.",
    "unknown tensor role %s":
        "A tensor role must be input, internal or output.",
    "duplicate stage name(s) %s":
        "Stage names key the task table; rename the repeated stages.",
    "stage %s binds register(s) %s twice":
        "One stage binds the same address register more than once.",
    "composer supports one to eight external inputs":
        "The composer's tensor table tops out at eight external inputs.",
    "late_inputs names unknown external input(s) %s":
        "late_inputs references a tensor that is not an external input of the graph.",
    "stage %s names unknown tensor %s":
        "A stage binding names a tensor outside the stage's declared tensors.",
    "no stage writes internal tensor(s) %s":
        "An internal tensor has no producer; drop it or add the stage that writes it.",
    "external tensor %s overlaps internal %s":
        "An external tensor's arena range collides with an internal tensor's range.",
    "stage %s register %#x out of range":
        "A computed register value does not fit in 32 bits; check the emitter's arithmetic.",
    "binding derivation requires a v5 container":
        "Reading bindings back only works on a version-5 sequence.",
    "unknown task family %s":
        "The (word count, enable) pair is not one of the known task families.",
    "declared bindings repeat a stage name":
        "The declared binding list names the same stage twice.",
    "schedule %s does not match the declared stages":
        "The task schedule and the declared stage list disagree; rebuild the container.",
    "task %d matches %d declarations":
        "A container task matches a number of declarations other than one.",
    "stage %s declares unknown tensor %s":
        "A declared binding names a tensor the container does not define.",
    "task %d (%s) %s differ: declared %s, container %s":
        "The declared read/write view differs from the emitted register words.",
    # --- depthwise.py --------------------------------------------------------
    "depthwise sequence requires Conv[/Relu] -> depthwise Conv":
        "Only Conv[/Relu] followed by one depthwise Conv lowers here.",
    "depthwise supports 1..16 channels":
        "The depthwise group (channel count) must be 1-16.",
    "depthwise supports stride 1 or 2":
        "Only depthwise stride 1 or 2 is lowered.",
    "depthwise input must be RGB, H/W 5..8":
        "The stem input must be [1,3,H,W] with H/W 5-8.",
    "depthwise kernels 1/3/5 only":
        "Only K1, K3 or K5 depthwise kernels are lowered.",
    "depthwise profile requires supported group/kernel/padding/stride and constant weights/bias":
        "The depthwise geometry or constants fall outside the supported profile.",
    "depthwise profile requires matching float32 weights [C,1,K,K] and bias [C]":
        "Weights must be [C,1,K,K] float32 and the bias [C] float32.",
    "depthwise profile requires matching static input/output shapes":
        "Input and output shapes must be static and consistent with kernel and stride.",
    "depthwise input must match group count at 8x8":
        "The stem output channel count must equal the depthwise group count.",
    "depthwise profile supports only 1x1 or 3x3 stem convolution":
        "The stem Conv must be 1x1 or 3x3.",
    "C5..16 requires a 1x1 stem":
        "Hidden widths above four force a 1x1 stem.",
    "stem weights exceed depthwise profile allocation":
        "The stem weight block is too large for the fixed 0x880 allocation.",
    "depthwise-pointwise requires Conv[/Relu] -> depthwise Conv -> pointwise Conv":
        "Only that three-stage shape lowers on the depthwise-pointwise path.",
    "bounded pointwise successor requires dense 1x1 constant weights":
        "The successor must be a dense 1x1 Conv with constant weights.",
    "pointwise successor shape mismatch":
        "The pointwise output shape does not extend the depthwise output as declared.",
    # --- depthwise_join.py ---------------------------------------------------
    "depthwise join requires default-domain nodes":
        "Every node in the depthwise join must use the default ONNX domain.",
    "depthwise join requires stem[,Relu], dense Conv, depthwise Conv and one join":
        "The graph must be stem[,Relu] plus a dense Conv, a depthwise Conv and one join.",
    "depthwise join requires a dense Conv, a depthwise Conv and Add/Mul/Sub/Max":
        "The visible branch nodes must be those two Convs and one join operator.",
    "depthwise join must not carry attributes":
        "The join operator must be attribute-free.",
    "depthwise join branches must both consume the shared stem output":
        "Both branches must read the stem output directly.",
    "depthwise join must consume the dense branch then the depthwise branch":
        "Join operand order chooses the buffer roles; pass the dense branch first.",
    "depthwise join requires one input and one output":
        "The depthwise join graph must be single-input, single-output.",
    "depthwise join tensors must be float32 [1,3,8,8]":
        "The depthwise join geometry is fixed at 8x8 C3.",
    "depthwise join external tensors must be float32":
        "External tensors must be float32.",
    "depthwise join convolutions require constant float32 weights and bias":
        "Both Convs need constant float32 weights and bias.",
    "depthwise join constants must be float32":
        "Every initializer must be float32.",
    "depthwise join stem must be a 1x1 Conv 3->3":
        "The stem must be a 1x1 Conv from 3 to 3 channels.",
    "unsupported stem attributes":
        "The stem Conv carries an attribute outside the profile's allowed set.",
    "depthwise join stem Relu must consume the stem output":
        "The optional Relu must directly follow the stem.",
    "depthwise join dense branch must be a 3-output 1x1/3x3 Conv":
        "The dense branch is a 3-output K1 or symmetric-pad K3 Conv.",
    "unsupported dense branch attributes":
        "The dense branch Conv has an attribute outside the allowed set.",
    "depthwise join 3x3 dense branch requires symmetric pad1":
        "A K3 dense branch needs pads [1,1,1,1].",
    "depthwise join branch must be a group3 depthwise 1x1/3x3/5x5 Conv":
        "The depthwise branch is group-3 K1, K3 or K5.",
    "unsupported depthwise branch attributes":
        "The depthwise branch Conv has an attribute outside the allowed set.",
    "depthwise join requires zero-centered join operands":
        "Both join operands must use zero point 0.",
    "the depthwise join output override requires a Mul join":
        "Only the Mul join can carry the output-scale override.",
    "depthwise join external tensor %s overlaps %s":
        "The external tensor's arena range collides with another tensor.",
    # --- elementwise.py ------------------------------------------------------
    "invalid Mul output quantization":
        "The requested Mul output scale or zero point is out of range.",
    "Mul output scale outside conversion range":
        "The product scale has no valid multiplier/shift pair; adjust the operands.",
    "Mul-to-Relu profile requires a terminal Relu":
        "The graph must end in Mul -> Relu for this fusion.",
    "Mul-to-Clip profile requires terminal constant Clip[0,6]":
        "The graph must end in Mul -> Clip with constant bounds 0 and 6.",
    "Mul Clip[0,6] currently uses output scale 6/255 and zero point -128":
        "The Mul+Clip fusion has one fixed output range; match it or drop the override.",
    "Mul Clip currently requires batch1 without mutable descriptors":
        "Batch>1 and mutable constants are not supported with the Mul+Clip fusion.",
    "Mul-to-Add profile requires a terminal scalar constant Add":
        "The graph must end in Mul -> Add with a scalar constant.",
    "Mul-to-Add requires a finite float32 scalar and explicit output quantization":
        "The folded Add needs a finite float32 scalar and an explicit output range.",
    "scalar Add must be exactly representable without overflowing the internal zero point":
        "The Add step must land on an integer code inside [-128,127].",
    "residual Add requires Conv(input) + matching RGB input":
        "A residual add needs Conv(image) plus a second RGB input of the same shape.",
    "external-plus-intermediate Mul requires Conv(input0) * matching RGB input1":
        "The second Mul operand must be an RGB external input matching input0.",
    "elementwise profile requires two Conv branches feeding Add, Mul, Sub or Max":
        "Only two independent Conv[/Relu] branches and one Add/Mul/Sub/Max lower here.",
    "invalid branch Relu":
        "A branch Relu carries attributes or more than one input.",
    "elementwise operands must come from Conv[/Relu] branches":
        "Both join operands must be Conv or Conv -> Relu outputs.",
    "elementwise branches must be independent and ordered":
        "The branch nodes must not share hidden tensors and must appear in graph order.",
    "elementwise profile requires two same-input Conv branches without broadcasting":
        "Both branches must read the graph inputs without broadcasting.",
    "elementwise requires RGB input, 2..16 output channels, H/W 5..8":
        "The elementwise geometry is RGB H/W 5-8 with 2-16 output channels.",
    "external branch inputs must have matching RGB shapes":
        "Every external branch input must share the RGB shape.",
    "elementwise branch shapes must agree; broadcasting unsupported":
        "Branch outputs and the graph output must share one shape.",
    "elementwise profile requires 1x1 Conv branches":
        "The elementwise branch Convs must be 1x1.",
    "Mul operand zero points must be two INT8 values":
        "Pass exactly two Mul operand zero points, each in [-128,127].",
    "operand zero points apply only to Mul":
        "Add, Sub and Max take no operand zero points.",
    "elementwise branch weight allocation exceeded":
        "The branch weight block exceeds the 64-byte slot the join buffer reserves.",
    "unexpected elementwise register layout":
        "REGISTERS no longer yields the 78 elementwise words the emitter assumes.",
    "standalone Mul requires one or two external inputs with matching RGB shapes":
        "A lone Mul needs one or two external RGB inputs of the same shape.",
    "standalone Mul requires C1..16":
        "The standalone Mul channel count must be 1-16.",
    "constant Mul requires one external input and one immutable constant":
        "One Mul operand must be an external input and the other a constant initializer.",
    "finite float32 broadcast constant and rank-four input required":
        "The constant must be finite float32 and the input rank-4.",
    "unsupported Mul broadcast shape":
        "The constant cannot broadcast to the input shape.",
    "single-batch constant Mul requires RGB H/W5..8":
        "Batch-1 constant Mul is fixed at RGB H/W 5-8.",
    "constant Mul uses a materialized zero-point-zero operand":
        "Only zero-centered constant operands are supported on this path.",
    "per-channel constant Mul requires one Mul with one external input":
        "The graph must be a single Mul against one initializer.",
    "per-channel constant Mul requires one immutable constant operand":
        "The other Mul operand must be a constant initializer.",
    "finite float32 constant and rank-four input required":
        "The constant must be finite float32 and the input rank-4.",
    "per-channel constant Mul requires single-batch RGB H/W5..8":
        "Per-channel constant Mul is fixed at batch 1, RGB H/W 5-8.",
    "per-channel constant Mul requires a broadcastable constant":
        "The constant must broadcast to the input shape.",
    "per-channel constant Mul requires a spatially invariant constant":
        "The constant must be per-channel [N,C,1,1].",
    "per-channel constant Mul requires distinct channel values":
        "At least two per-channel constant values must differ.",
    "runtime scale Mul requires one Mul with two external inputs":
        "The graph must be one Mul whose both operands are external inputs.",
    "runtime scale Mul requires [1,3,H,W] and [1,3,1,1] external inputs":
        "Supply the image and a per-channel scale of exactly those shapes.",
    "runtime scale Mul requires H/W 5..8":
        "The runtime-scale image must be H/W 5-8.",
    "operand scale must be a positive finite number":
        "Pass a positive finite operand scale.",
    "unexpected elementwise payload size":
        "The reused Mul payload is not the expected 4096 bytes.",
    "batched constant Mul requires N2..16, C1..16, H/W1..32 and zero-centered constant arithmetic":
        "Batched constant Mul bounds are N2-16, C1-16, H/W1-32 and zero-centered operands.",
    "batched constant Mul supports scalar or [N,C,1,1] constants":
        "Only scalar or per-batch/channel constants broadcast in the batched path.",
    # --- elementwise_chain.py ------------------------------------------------
    "elementwise DAG requires Conv, Conv, {Add|Mul|Sub|Max}, Mul...":
        "The chain must start with two Convs and one join, then continue with Mul stages.",
    "unsupported operator domain":
        "Every node in the graph must use the default ONNX domain.",
    "elementwise DAG requires two inputs and one output":
        "The elementwise DAG is two-input, one-output.",
    "the first stage must combine the two Conv branches":
        "The third node must join the two Conv outputs.",
    "each later stage must multiply the previous stage by the first branch":
        "Later stages must be Mul(previous_stage, first_branch).",
    "the last stage must produce the graph output":
        "The final node must write the graph output.",
    "Conv branches must consume the two graph inputs in order":
        "Branch 0 reads input 0 and branch 1 reads input 1.",
    "elementwise DAG operators take no attributes":
        "Add/Mul/Sub/Max in the DAG must be attribute-free.",
    "elementwise DAG branches must be 1x1 Conv":
        "Only 1x1 Conv branches are lowered.",
    "elementwise DAG branches require constant weights and bias":
        "Both Convs need constant float32 weights and bias.",
    "elementwise DAG external tensors must be float32 [1,3,8,8]":
        "The elementwise DAG boundary is float32 [1,3,8,8].",
    "missing static first-stage output shape":
        "The first-stage output shape is not static; run shape inference first.",
    "unexpected first-stage container":
        "The compiled first stage is not the expected one-task container.",
    "elementwise DAG requires zero-point-zero stages":
        "Every DAG stage must run at zero point 0.",
    "elementwise DAG requires the verified equal-scale ":
        "The measured branch scales must match the verified equal-scale band.",
    "unknown first stage: ":
        "The first join operator is not one of Add, Mul, Sub or Max.",
    # --- elementwise_multi.py ------------------------------------------------
    "multi-input DAG requires Conv, Conv, {Add|Mul|Sub|Max}, (Conv, Mul)*":
        "The multi-input DAG is Conv, Conv, one join, then (Conv, Mul) pairs.",
    "each extra input requires a Conv then a Mul":
        "Every additional input contributes a Conv followed by a Mul.",
    "multi-input DAG requires one input per Conv and one output":
        "The input count must match the Conv count; the DAG has one output.",
    "each extra stage must multiply the previous result by its Conv":
        "Each later stage must be Mul(previous_result, that stage's Conv output).",
    "Conv branches must consume the graph inputs in order":
        "The Convs must read the graph inputs in declaration order.",
    "multi-input DAG requires zero-point-zero stages":
        "Every multi-input DAG stage must run at zero point 0.",
    "multi-input DAG requires the verified equal-scale ":
        "The measured branch scales must match the verified equal-scale band.",
    # --- graph.py ------------------------------------------------------------
    "two-head profile requires Conv, Relu, Conv, Conv":
        "The two-head profile is Conv -> Relu -> Conv -> Conv.",
    "two-head profile requires a valid stem Relu":
        "The stem Relu must directly follow the stem Conv and carry no attributes.",
    "two-head profile requires both heads to consume the stem output":
        "Both head Convs must read the stem output.",
    "two-head profile requires one input and two outputs":
        "The two-head graph needs one image input and two head outputs.",
    "two-head outputs must be the two head outputs in order":
        "Declare the two head outputs in head order.",
    "two-head external tensors must be float32 [1,3,8,8]":
        "The two-head boundary is float32 [1,3,8,8].",
    "two-head convolutions require constant float32 weights and bias":
        "All three Convs need constant float32 weights and bias.",
    "two-head constants must be float32":
        "Every initializer in the two-head graph must be float32.",
    "two-head stem must be a 1x1 Conv with hidden channels 3..16":
        "The stem is a 1x1 Conv to 3-16 hidden channels.",
    "unsupported head attributes":
        "A head Conv carries an attribute outside the profile's allowed set.",
    "two-head heads must be dense 3-output 1x1/3x3 Conv with hidden inputs":
        "Heads are 3-output K1 or symmetric-pad K3 Convs reading the stem output.",
    "two-head 3x3 heads require symmetric pad1":
        "K3 heads need pads [1,1,1,1].",
    "diamond profile requires default-domain nodes":
        "Every diamond node must use the default ONNX domain.",
    "diamond profile requires stem[,Relu], two heads and one join":
        "The diamond is stem[,Relu] plus two heads and one join.",
    "diamond join must be Add, Mul, Sub or Max without attributes":
        "The diamond join must be an attribute-free Add/Mul/Sub/Max.",
    "diamond tail must be [Conv, Relu]* Conv":
        "The optional tail is Conv/Relu alternating and ends in Conv.",
    "diamond tail Relu must not carry attributes":
        "Tail Relu nodes must be attribute-free.",
    "diamond stem must be Conv or Conv,Relu":
        "The diamond stem is one Conv with an optional Relu.",
    "diamond heads must be Conv":
        "Both diamond heads must be Conv nodes.",
    "diamond stem Relu must consume the stem output":
        "The stem Relu must read the stem Conv output.",
    "diamond heads must consume the stem output":
        "Both heads must read the stem output.",
    "diamond join must consume both head outputs in order":
        "The join takes head 0 then head 1.",
    "diamond tail must consume the join output":
        "The tail's first node must read the join output.",
    "diamond tail nodes must be connected in order":
        "Each tail node must consume the previous tail node's output.",
    "diamond profile requires one input and one output":
        "The diamond graph is single-input, single-output.",
    "diamond external tensors must be float32 [1,3,8,8]":
        "The diamond boundary is float32 [1,3,8,8].",
    "diamond heads must produce float32 [1,3,8,8]":
        "Head outputs must be float32 [1,3,8,8].",
    "diamond tail must produce float32 [1,3,8,8]":
        "Tail outputs must be float32 [1,3,8,8].",
    "diamond convolutions require constant float32 weights and bias":
        "Every diamond Conv needs constant float32 weights and bias.",
    "diamond constants must be float32":
        "Every diamond initializer must be float32.",
    "diamond stem must be a 1x1 Conv with hidden channels 3..16":
        "The stem is a 1x1 Conv to 3-16 hidden channels.",
    "diamond heads must be dense 3-output 1x1/3x3 Conv with hidden inputs":
        "Heads are 3-output K1 or symmetric-pad K3 Convs.",
    "diamond 3x3 heads require symmetric pad1":
        "K3 diamond heads need pads [1,1,1,1].",
    "operand zero points apply only to the Mul join":
        "Add, Sub and Max joins take no operand zero points.",
    "the diamond output override requires the Mul join or a Conv tail":
        "Only a Mul join or a Conv tail can carry the diamond output override.",
    "diamond tail layers must be dense 3-output 1x1/3x3 Conv with three inputs":
        "Tail layers are 3-input, 3-output K1 or symmetric-pad K3 Convs.",
    "unsupported tail attributes":
        "A tail Conv carries an attribute outside the profile's allowed set.",
    "diamond 3x3 tail layers require symmetric pad1":
        "K3 tail layers need pads [1,1,1,1].",
    "diamond join requires adjacent head buffers":
        "The two head surfaces must be adjacent for the depthwise/elementwise read.",
    "join chain requires default-domain nodes":
        "Every join-chain node must use the default ONNX domain.",
    "join chain requires stem[,Relu] then head,Mul pairs":
        "The join chain is stem[,Relu] followed by head/Mul pairs.",
    "chained Mul joins require zero-centered operands":
        "Every chained Mul join must use zero point 0.",
    "join chain supports 3..8 heads":
        "The join chain folds 3-8 heads; split wider graphs.",
    "join chain stem Relu must consume the stem output":
        "The stem Relu must read the stem Conv output.",
    "join chain heads must be Conv nodes reading the stem output":
        "Every head must be a Conv reading the stem output.",
    "join chain joins must be attribute-free Add/Mul/Sub/Max nodes":
        "The join nodes must be attribute-free Add/Mul/Sub/Max.",
    "join chain joins must consume the previous result and the next head in order":
        "Each join takes the running result then the next head.",
    "join chain tail Relu must not carry attributes":
        "Tail Relu nodes must be attribute-free.",
    "join chain tail must consume the last join output":
        "The tail's first node must read the last join output.",
    "join chain tail nodes must be connected in order":
        "Each tail node must consume the previous tail node's output.",
    "join chain runtime tail must combine the last join result with one external input":
        "The runtime residual tail must be a Mul of the running result and one external input.",
    "join chain requires one image input and one output":
        "The join chain is one image input and one output.",
    "join chain external tensors must be float32 [1,3,8,8]":
        "The join-chain boundary is float32 [1,3,8,8].",
    "join chain runtime tail requires a float32 %s input":
        "The runtime-tail operand must be a float32 tensor of the shape the message names.",
    "join chain heads must produce float32 [1,3,8,8]":
        "Join-chain heads must produce float32 [1,3,8,8].",
    "join chain tail must produce float32 [1,3,8,8]":
        "The join-chain tail must produce float32 [1,3,8,8].",
    "join chain convolutions require constant float32 weights and bias":
        "Every join-chain Conv needs constant float32 weights and bias.",
    "join chain constants must be float32":
        "Every join-chain initializer must be float32.",
    "join chain stem must be a 1x1 Conv with hidden channels 3..16":
        "The stem is a 1x1 Conv to 3-16 hidden channels.",
    "join chain heads must be dense 3-output 1x1/3x3 Conv with hidden inputs":
        "Heads are 3-output K1 or symmetric-pad K3 Convs.",
    "join chain 3x3 heads require symmetric pad1":
        "K3 join-chain heads need pads [1,1,1,1].",
    "join chain depthwise heads require a three-channel stem and group3 1x1/3x3/5x5 weights":
        "Depthwise heads need a C3 stem and group-3 K1/K3/K5 weights.",
    "unsupported depthwise head attributes":
        "A depthwise head Conv carries an attribute outside the allowed set.",
    "join chain tail layers must be dense 3-output 1x1/3x3 Conv with three inputs":
        "Tail layers are 3-input, 3-output K1 or symmetric-pad K3 Convs.",
    "join chain 3x3 tail layers require symmetric pad1":
        "K3 tail layers need pads [1,1,1,1].",
    "the join chain output override requires a Mul join or a Conv tail":
        "Only a Mul join or a Conv tail can carry the join-chain output override.",
    "the join chain output override is unsupported for a runtime residual tail":
        "A runtime residual tail cannot take an output-scale override.",
    "join chain external tensor %s overlaps %s":
        "The external tensor's arena range collides with another tensor.",
    # --- join_dag.py ---------------------------------------------------------
    "join DAG requires a 1x1 stem, three branches and two joins over produced tensors":
        "The DAG is a 1x1 stem, three branches and two joins of produced tensors.",
    "join DAG requires one input, one output and a final join or pool":
        "The join DAG is one input and one output ending in a join or a pool.",
    "join DAG external tensors must be float32 %s":
        "The external tensors must be float32 with the shape the message names.",
    "join DAG terminal pool must be 2x2 stride-2 MaxPool or AveragePool":
        "The terminal pool must be 2x2 stride-2 MaxPool or AveragePool.",
    "join DAG branch outputs must be float32 1xCx8x8 tensors":
        "Every branch output must be a float32 1xCx8x8 tensor.",
    "join DAG stem must be a 1x1 Conv with hidden channels 3..16":
        "The DAG stem is a 1x1 Conv to 3-16 hidden channels.",
    "join DAG stem Relu must consume the stem output":
        "The stem Relu must read the stem Conv output.",
    "join DAG branch layers must chain off the previous layer":
        "Each branch layer must consume the previous layer in its branch.",
    "join DAG branch layers require constant float32 weights and bias":
        "Every branch Conv needs constant float32 weights and bias.",
    "join DAG chained depthwise layers need group C, 1x1/3x3 and symmetric pad":
        "Chained depthwise layers are group-C K1/K3 with symmetric padding.",
    "join DAG dense layers must be C1..16 1x1/3x3 Conv over the previous tensor":
        "Dense branch layers are C1-16 K1 or symmetric-pad K3 Convs.",
    "unsupported dense layer attributes":
        "A dense branch Conv carries an attribute outside the allowed set.",
    "join DAG depthwise branches must be a single layer":
        "A depthwise branch in the DAG must be exactly one layer.",
    "join DAG depthwise branches require a three-channel stem and group3 weights":
        "Depthwise branches need a C3 stem and group-3 weights.",
    "unsupported depthwise layer attributes":
        "A depthwise branch Conv carries an attribute outside the allowed set.",
    "join DAG branch outputs must have three channels":
        "Every branch must end with three output channels.",
    "join DAG joins may only consume three-channel tensors":
        "The joins only accept three-channel operands.",
    "join DAG Add/Sub/Max needs both operands on one scale; reused tensors fixed different bands":
        "Both Add/Sub/Max operands must share one output band; rebuild the reused tensor.",
    "join DAG cannot re-quantize a join result onto another band":
        "A join result cannot be re-quantized; restructure the DAG.",
    "join DAG operand was already committed to another band":
        "A reused operand was already pinned to a different band; split the tensor.",
    "join DAG requires default-domain nodes":
        "Every join-DAG node must use the default ONNX domain.",
    "join DAG external tensor %s overlaps %s":
        "The external tensor's arena range collides with another tensor.",
    # --- layout.py -----------------------------------------------------------
    "terminal Reshape requires an immutable shape and one output":
        "The terminal Reshape needs a constant shape initializer and one output.",
    "Reshape must preserve batch, channels and spatial pixel count":
        "Only Reshapes that keep batch, channels and pixel count are supported.",
    # --- liveness.py ---------------------------------------------------------
    "schedule requires at least one task":
        "The allocator needs at least one task to plan.",
    "task %d must be Access or {'reads','writes'}":
        "Describe each task as an Access or a dict with reads and writes.",
    "task %d tensor names must be non-empty strings":
        "A task read/write tensor name is empty or not a string.",
    "task %d writes a tensor twice":
        "A task lists the same output tensor twice; deduplicate it.",
    "task %d reads its own output tensor":
        "A task cannot read a tensor it writes in the same task.",
    "tensor %s has more than one producer":
        "Two tasks write the same tensor; make the producer unique.",
    "tensor %s is both defined and written":
        "An external tensor is also written by a task; pick one role.",
    "tensor %s is read before it is written":
        "A task reads a tensor before any producer writes it; fix the schedule order.",
    "task graph contains a cycle":
        "The read/write graph is cyclic; the allocator needs a DAG.",
    "missing live intervals for tensors: %s":
        "No live interval was computed for a tensor; add it to the schedule.",
    "tensor %s needs a positive size":
        "A tensor's arena size is zero or negative.",
    "no arena placement for tensor %s":
        "The allocator could not place a tensor; reduce sizes or enable reuse.",
    "tensor %s is not %d-byte aligned":
        "A tensor offset violates the arena alignment; realign it.",
    "tensors %s and %s overlap while live":
        "Two simultaneously live tensors share arena bytes; disable reuse or resize.",
    # --- lut.py --------------------------------------------------------------
    "LUT profile requires Conv followed by Sigmoid or Tanh":
        "The LUT profile is Conv -> Sigmoid or Conv -> Tanh.",
    "bounded LUT profile requires a C3 1x1 Conv stem, 8x8, input scale1 zero point128":
        "The LUT stem must be C3 1x1 at 8x8 with input scale 1 and zero point 128.",
    "LUT stem weights and bias must be finite":
        "The LUT stem weights/bias contain NaN or infinity.",
    "LUT stem must be a diagonal 1x1 Conv with one scalar magnitude per channel":
        "The stem must be a diagonal 1x1 Conv with one scalar magnitude per channel.",
    "LUT stem channels must share one negative-half gain band; got %s":
        "All stem channels must share one negative-half gain band; the message shows the bands found.",
    ("LUT stem weight scale must make BASE_WEIGHT_SCALE/scale a power of two; other bands "
     "advance the negative table half by whole bits and leave \u00b11 rounding"):
        "A power-of-two scale ratio is required so the negative table half stays within \u00b11 rounding.",
    "LUT stem output range [%.3f, %.3f] leaves the table domain [%.3f, %.3f]":
        "The stem output range leaves the table domain; widen the gains or narrow the table.",
    "unexpected LUT setup length":
        "The emitted LUT setup is not the expected 1106 words.",
    # --- model.py ------------------------------------------------------------
    "truncated model header":
        "The container file is shorter than its header; re-emit it.",
    "unsupported model format":
        "The container format version is not one this reader supports.",
    "unsupported NPU target/profile":
        "The container targets a different NPU profile than the reader implements.",
    "unsupported pooling profile":
        "The container's pooling profile is not implemented by this reader.",
    "unsupported two-layer profile":
        "The container's two-layer profile is not implemented by this reader.",
    "unsupported tensor shape":
        "The container declares a shape outside the supported bounds.",
    "unsupported memory layout":
        "The container uses a memory layout this reader does not implement.",
    "unsupported model options":
        "The container sets option bits the reader does not understand.",
    "nonzero reserved fields":
        "A reserved header field is nonzero; the file is corrupt or from another build.",
    "invalid input quantization":
        "The input scale or zero point is out of range.",
    "incorrect model length":
        "The declared length does not match the file; the container is truncated or padded.",
    "model checksum mismatch":
        "The container checksum failed; the file was modified or corrupted.",
    # --- native.py -----------------------------------------------------------
    "Clip upper clamp outside INT32 accumulator range":
        "The computed Clip[0,6] upper clamp does not fit INT32; rescale the layer.",
    "native input profile requires one Conv":
        "The native input profile is one Conv with an optional Relu or Clip.",
    "invalid native Relu connection":
        "The Relu must be the standard node directly fed by the Conv.",
    "native Clip fusion requires constant scalar range [0,6]":
        "The fused Clip must use constant scalar bounds 0 and 6.",
    "native Conv requires static batch1..16, H/W1..128, input C1..16352, constant weights/bias":
        "Use a static batch 1-16, H/W 1-128, input C 1-16352 Conv with constant weights/bias.",
    "native Conv weights must have rank four":
        "Native Conv weights must be a rank-4 [O,I,K,K] tensor.",
    "native padding/stride/dilation unsupported":
        "Use pads 0-255, stride 1-4 and dilation 1-17.",
    "invalid native Conv output geometry":
        "The output geometry is not positive or the padding exceeds the kernel.",
    "native Conv supports odd K1..31, explicit padding, stride 1..4, input C1..16352/output C1..8192":
        "Only odd K1-31, explicit padding, stride 1-4 and C1-16352 in/C1-8192 out are lowered.",
    "invalid input zero point":
        "The input zero point must be an integer in [0,255].",
    "prequantized native Conv already carries output quantization":
        "A prequantized Conv cannot also take Clip/Relu/output-range options.",
    "native Conv geometry cannot be safely height-tiled":
        "The image exceeds the CNA atom budget and cannot be split into safe height tiles.",
    "native Conv geometry has no aligned height tiling":
        "No 64-byte-aligned height tiling fits the atom budget; change the geometry.",
    "native Conv height tiling made no progress":
        "The tiler could not advance a row; check the geometry bounds.",
    "native input channel tiling unsupported":
        "Input channels must align to 16-lane planes within the 511-part weight-table budget (C1-16352).",
    # --- native_elementwise.py -----------------------------------------------
    "native elementwise requires two Conv branches":
        "The native elementwise profile is two Conv[/Relu] branches and one join.",
    "invalid native elementwise graph connections":
        "The branch wiring does not match the supported native elementwise graph.",
    "native elementwise requires matching static input shapes":
        "Both branches must read inputs with matching static shapes.",
    "missing branch shape":
        "A branch output shape is missing; run shape inference first.",
    "native elementwise branches require same-shape 1x1 Conv":
        "Both branches must be same-shape 1x1 Convs.",
    # --- network.py ----------------------------------------------------------
    "expected Conv-Relu-Conv followed by three pools":
        "The legacy network profile is Conv-Relu-Conv plus three pools.",
    "three matching 2x2 stride-2 pools required":
        "All three pools must be matching 2x2 stride-2 pools.",
    "network output must be float32 [1,3,1,1]":
        "The network output must be float32 [1,3,1,1].",
    "network constants exceed memory layout":
        "The legacy network constants do not fit the fixed memory layout.",
    # --- padding.py ----------------------------------------------------------
    "unsupported Pad mode: ":
        "Only the supported Pad modes lower; the message names the mode found.",
    "pad_input expects HWC or NHWC data":
        "Pass rank-3 HWC or rank-4 NHWC data to the padding helper.",
    "negative padding is not supported":
        "Padding amounts must be non-negative.",
    # --- pool_join.py --------------------------------------------------------
    "pool join requires default-domain nodes":
        "Every pool-join node must use the default ONNX domain.",
    "pool join requires stem[,Relu], Conv, pool, Conv, pool and one join":
        "The graph is stem[,Relu] plus two Conv/pool branches and one join.",
    "pool join requires two Conv branches and Add/Mul/Sub/Max":
        "The branches must end in Convs and the join must be Add/Mul/Sub/Max.",
    "pool join requires two matching MaxPool or AveragePool nodes":
        "Both branches must carry matching MaxPool or AveragePool nodes.",
    "pool join supports 2x2 stride-2 pooling without extra attributes":
        "Only 2x2 stride-2 pools with no extra attributes are lowered.",
    "pool join must not carry attributes":
        "The pool-join operator must be attribute-free.",
    "pool join branches must run stem -> Conv -> pool and join both pools in order":
        "Each branch must run stem -> Conv -> pool, and the join takes pool 0 then pool 1.",
    "pool join requires one input and one output":
        "The pool join is single-input, single-output.",
    "pool join tensors must be float32 [1,3,8,8] or [1,3,4,4]":
        "Pool-join tensors must be float32 [1,3,8,8] or the pooled [1,3,4,4].",
    "pool join external tensors must be float32":
        "External tensors must be float32.",
    "pool join convolutions require constant float32 weights and bias":
        "Both Convs need constant float32 weights and bias.",
    "pool join constants must be float32":
        "Every pool-join initializer must be float32.",
    "pool join stem must be a 1x1 Conv with hidden channels 3..16":
        "The stem is a 1x1 Conv to 3-16 hidden channels.",
    "pool join stem Relu must consume the stem output":
        "The stem Relu must read the stem Conv output.",
    "pool join branches must be dense 3-output 1x1/3x3 Conv with hidden inputs":
        "Both branches are 3-output K1 or symmetric-pad K3 Convs.",
    "unsupported branch attributes":
        "A pool-join branch Conv carries an attribute outside the allowed set.",
    "pool join 3x3 branches require symmetric pad1":
        "K3 pool-join branches need pads [1,1,1,1].",
    "the pool join output override requires a Mul join":
        "Only the Mul join can carry the pool-join output override.",
    # --- pooled_branches.py --------------------------------------------------
    "pooled branches require a 1x1 stem, two or three Conv-chain branches with pools and one or two joins":
        "Pooled branches are a 1x1 stem, two or three pooled Conv chains and one or two joins.",
    "pooled branches require one input, one output and a final join":
        "The pooled-branches graph is one input, one output and ends in a join.",
    "pooled branches external tensors must be float32 %s":
        "External tensors must be float32 with the shape the message names.",
    "pooled branches branch tensors must be float32 1xCx8x8":
        "Branch tensors must be float32 1xCx8x8.",
    "pooled branches pools must be 2x2 stride-2 MaxPool/AveragePool":
        "Pools must be 2x2 stride-2 MaxPool or AveragePool.",
    "pooled branches stem must be a 1x1 Conv with hidden channels 3..16":
        "The stem is a 1x1 Conv to 3-16 hidden channels.",
    "pooled branches stem Relu must consume the stem output":
        "The stem Relu must read the stem Conv output.",
    "pooled branches layers must chain off the previous layer":
        "Each branch layer must consume the previous layer in its branch.",
    "pooled branches layers require constant float32 weights and bias":
        "Every branch Conv needs constant float32 weights and bias.",
    "pooled branches chained depthwise layers need group C, 1x1/3x3 and symmetric pad":
        "Chained depthwise layers are group-C K1/K3 with symmetric padding.",
    "pooled branches layers must be C1..16 1x1/3x3 Conv over the previous tensor":
        "Branch layers are C1-16 K1 or symmetric-pad K3 Convs.",
    "unsupported layer attributes":
        "A pooled-branch Conv carries an attribute outside the allowed set.",
    "pooled branches layers must be dense or depthwise group C":
        "Branch layers must be dense Convs or group-C depthwise Convs.",
    "pooled branches must end each chain with three channels":
        "Every pooled branch must end with three channels.",
    "pooled branches joins must fold the pooled branches in order":
        "The joins must fold the pooled branches in declaration order.",
    "pooled branches require default-domain nodes":
        "Every pooled-branches node must use the default ONNX domain.",
    "pooled branches require matching pool kinds":
        "All branches must use the same pooling kind.",
    # --- pooling.py ----------------------------------------------------------
    "unsupported pooling graph":
        "Only Conv[/Relu] -> 2x2 stride-2 MaxPool/AveragePool at 8x8 C3 lowers here.",
    # --- quantization.py -----------------------------------------------------
    "invalid UINT8 input quantization":
        "The UINT8 input needs a positive scale and a zero point in [0,255].",
    "expected [O,I,K,K], I=1 or 3, 1<=O<=16, K=1, 3 or 5 weights":
        "Quantization wants the legacy [O,I,K,K] weight profile.",
    "finite weights and bias required":
        "Weights and bias must contain only finite values.",
    "INT32 bias overflow":
        "The quantized bias exceeds INT32; widen the output scale.",
    "INT32 accumulator overflow":
        "A worst-case accumulator bound exceeds INT32; rescale the layer.",
    "invalid output quantization":
        "The output scale or zero point is out of range.",
    "output scale outside validated multiplier/shift range":
        "No valid multiplier/shift pair exists for this output scale.",
    "unknown rounding candidate":
        "The rounding candidate must be ties-even or ties-away-zero.",
    # --- quantized_import.py -------------------------------------------------
    "QLinearConv requires scalar activation quantization constants":
        "QLinearConv activation scales and zero points must be scalar constants.",
    "QLinearConv INT32 accumulator overflow":
        "The QLinearConv accumulator can exceed INT32; rescale the layer.",
    "source weight scales exceed native per-channel conversion range":
        "A source weight scale is outside the native per-channel Q14 range.",
    "source output scale exceeds native conversion range":
        "The source output scale has no valid native multiplier/shift.",
    "quantized import requires one standard QLinearConv":
        "Only a single standard QLinearConv node is imported.",
    "QLinearConv requires one activation input/output and constant parameters":
        "QLinearConv needs one activation input/output and constant weight/bias.",
    "QLinearConv weights must be constant INT8 with float32 scale and INT8 zero point":
        "Weights must be constant INT8, the scale float32 and the zero point INT8.",
    "weight quantization must be scalar or per output channel":
        "Weight scale and zero point must be scalar or per output channel.",
    "QLinearConv scales must be finite and positive":
        "Every QLinearConv scale must be finite and positive.",
    "QLinearConv bias must be constant INT32 [output_channels]":
        "Bias must be a constant INT32 vector with one value per output channel.",
    "Q/DQ import requires input DQ, constant-weight DQ, Conv, output Q":
        "The graph must be input DequantizeLinear -> Conv -> QuantizeLinear with a constant-weight DQ.",
    "invalid Q/DQ Conv graph connections":
        "The Q/DQ nodes do not wrap the Conv the way the importer expects.",
    "unsupported Q/DQ attributes":
        "A Q/DQ node carries an attribute outside the supported set.",
    "Q/DQ weights must be constant INT8/float32/INT8":
        "Weights must be constant INT8 with float32 scale and INT8 zero point.",
    "per-channel weight Q/DQ requires axis0":
        "Per-channel weight quantization must use axis 0.",
    "Q/DQ scales must be finite and positive":
        "Every Q/DQ scale must be finite and positive.",
    "Q/DQ Conv bias must be constant float32 [output_channels]":
        "Bias must be constant float32 with one value per output channel.",
    "Q/DQ Conv bias is outside INT32 accumulator range":
        "The rounded bias does not fit INT32; rescale the layer.",
    # --- reduction.py --------------------------------------------------------
    "unsupported staged pooling graph":
        "Only the supported Conv + 2x2 stride-2 staged pooling chain lowers.",
    "staged pooling requires float32 output":
        "The staged pool output must be float32.",
    # --- scheduler.py --------------------------------------------------------
    "batched submission rejected: ":
        "The requested batched submission failed validation; the suffix names the reason.",
    "leading Pad requires constant pads":
        "A leading Pad needs a constant pads initializer.",
    "leading Pad with explicit axes is not supported":
        "The leading Pad must not carry an axes input.",
    "leading Pad must pad a rank-four NCHW input":
        "The leading Pad must pad a rank-4 NCHW input.",
    "negative Pad amounts are not supported":
        "Pad amounts must be non-negative.",
    "leading Pad requires a static NCHW input":
        "The padded input shape must be static NCHW.",
    "submission must be None, serial or batched":
        "Pass --submission serial or --submission batched.",
    "tiles must be an integer of at least 2":
        "--tiles must be an integer of at least 2.",
    "calibration ranges lack tensor ":
        "The calibration report is missing a tensor the scheduler needs.",
    "mutable weights currently require one native Conv[/Relu]":
        "Mutable weights only work with one native Conv[/Relu].",
    "mutable constants currently require one constant Mul":
        "Mutable constants only work with one constant-Mul profile.",
    "Mul operand zero points require a Mul profile":
        "Per-operand zero points are only valid for a Mul profile.",
    "QLinearConv carries its own quantization parameters":
        "QLinearConv supplies its own scales; do not pass overrides.",
    "Q/DQ Conv carries its own quantization parameters":
        "The Q/DQ Conv supplies its own scales; do not pass overrides.",
    "join chain requires the established UINT8 scale1/zero-point0 boundary":
        "The join chain only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "calibration is unsupported for the join chain":
        "Calibration is not supported for the join chain; use its fixed boundary.",
    "join DAG requires the established UINT8 scale1/zero-point0 boundary":
        "The join DAG only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "calibration is unsupported for the join DAG":
        "Calibration is not supported for the join DAG; use its fixed boundary.",
    "depthwise join requires the established UINT8 scale1/zero-point0 boundary":
        "The depthwise join only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "calibration is unsupported for the depthwise join":
        "Calibration is not supported for the depthwise join; use its fixed boundary.",
    "pool join requires the established UINT8 scale1/zero-point0 boundary":
        "The pool join only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "calibration is unsupported for the pool join":
        "Calibration is not supported for the pool join; use its fixed boundary.",
    "pooled branches require the established UINT8 scale1/zero-point0 boundary":
        "Pooled branches only accept the UINT8 scale 1 / zero point 0 boundary.",
    "calibration is unsupported for pooled branches":
        "Calibration is not supported for pooled branches; use their fixed boundary.",
    "output override unsupported for LUT profile":
        "The LUT profile derives its own output band; drop the output override.",
    "output override unsupported for LeakyRelu profile":
        "The LeakyRelu profile derives its own band; drop the output override.",
    "output override unsupported for PRelu profile":
        "The PRelu profile derives its own band; drop the output override.",
    "output override unsupported for Reshape profile":
        "The Reshape profile derives its own band; drop the output override.",
    "two-head profile requires the established UINT8 scale1/zero-point0 boundary":
        "The two-head profile only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "output override unsupported for the two-head profile":
        "The two-head profile derives its own band; drop the output override.",
    "diamond profile requires the established UINT8 scale1/zero-point0 boundary":
        "The diamond profile only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "native chain requires the established UINT8 scale1/zero-point0 boundary":
        "The native chain only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "height-strip tiling cannot be combined with output overrides, exposed intermediates or arena reuse":
        "Tiling is incompatible with output overrides, exposed intermediates and arena reuse.",
    "legacy Conv chain requires the established UINT8 scale1/zero-point0 boundary":
        "The legacy Conv chain only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "multi-input elementwise DAG requires the established UINT8 scale1/zero-point0 boundary":
        "The multi-input elementwise DAG only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "output override unsupported for the multi-input elementwise DAG":
        "The multi-input DAG derives its own band; drop the output override.",
    "elementwise DAG requires the established UINT8 scale1/zero-point0 boundary":
        "The elementwise DAG only accepts the UINT8 scale 1 / zero point 0 boundary.",
    "output override unsupported for the elementwise DAG":
        "The elementwise DAG derives its own band; drop the output override.",
    "sequence lowering requires one input, one output, and an initial Conv":
        "Sequence lowering needs one input, one output and a leading Conv.",
    "static NCHW input required":
        "The input shape must be static NCHW.",
    "output override unsupported for this strided profile":
        "The strided profile derives its own band; drop the output override.",
    "output override unsupported for this scheduled profile":
        "This scheduled profile derives its own band; drop the output override.",
    "sequence lowering currently supports Conv[/Relu] followed by 2x2 pooling":
        "The sequence scheduler lowers Conv[/Relu] plus 2x2 pooling only.",
    "missing static convolution output shape":
        "The Conv output shape is not static; run shape inference first.",
    "unsupported pooling attributes, shape, or graph connections":
        "The pooling node's attributes, shape or wiring are outside the profile.",
    "sequence output shape does not match graph":
        "The declared output shape disagrees with the graph; fix the model.",
    "too many NPU tasks":
        "The graph needs more tasks than the loader's 64-entry table; split it.",
        "v5 containers have no constant descriptor table; runtime replaceable parameters require a v4 container":
        "Version 5 has no constant table; use a v4 container (encode_sequence) for runtime-replaceable parameters.",
# --- sequence.py ---------------------------------------------------------
    "unknown tensor layout":
        "A tensor layout code is not packed-U8, native16 or packed-INT8.",
    "invalid tensor descriptor":
        "A version-5 tensor descriptor is malformed.",
    "tensor descriptor size mismatch":
        "A tensor's declared size does not match its layout and shape.",
    "v5 requires at least one external input and output":
        "A version-5 container needs at least one external input and one output.",
    "external tensor indices must be contiguous from zero":
        "External input/output indices must run 0, 1, 2, ...",
    "too many constant descriptors":
        "The container has more than the 64 constant-descriptor slots.",
    "invalid constant descriptor":
        "A version-4 constant descriptor is malformed.",
    "unknown input layout":
        "The header input-layout flag is not packed UINT8 or native16.",
    "batch must be 1..16":
        "The container batch must be 1-16.",
    "logical input tensor count unsupported":
        "Only the supported external-input counts encode in this container version.",
    "invalid v5 sequence header":
        "The version-5 header fields are inconsistent.",
    "truncated v5 extension":
        "The file ends inside the version-5 extension.",
    "invalid v5 extension":
        "A version-5 extension field is invalid.",
    "invalid task count":
        "The declared task count is outside the loader's table.",
    "invalid sequence allocation":
        "Payload, arena or IO offsets and sizes violate the layout.",
    "incorrect sequence length":
        "The declared payload length does not match the file.",
    "invalid sequence quantization":
        "A header scale or zero point field is out of range.",
    "invalid task descriptor":
        "A task descriptor's offset, count, enable or mask is inconsistent.",
    "invalid tensor name":
        "A tensor name is empty, too long or not NUL-terminated.",
    "tensor outside arena":
        "A tensor's byte range leaves the arena.",
    "overlapping external tensors":
        "Two external tensors share arena bytes; they must be disjoint.",
    "internal tensor overlaps external tensor":
        "An internal tensor overlaps an external one.",
    "v5 primary tensor mismatch":
        "The header's primary shape/offset does not match external tensor 0.",
    "sequence checksum mismatch":
        "The FNV-1a checksum failed; the file was modified or corrupted.",
    "truncated sequence header":
        "The file is shorter than the 96-byte sequence header.",
    "invalid sequence magic":
        "The file does not start with the ORNPUSEQ magic.",
    "invalid sequence header":
        "A header version or size field is unsupported.",
    "invalid two-input tensor layout":
        "The two-input layout does not match the declared input count.",
    "batched sequence requires native16 layout":
        "Batched submission needs the native16 input layout.",
    "invalid sequence shape":
        "A declared dimension is outside the loader's bounds.",
    "invalid stride/task count":
        "The input row stride or task count is inconsistent with the header.",
    "overlapping or out-of-bounds IO buffers":
        "Input/output extents leave the arena or overlap each other.",
    "invalid constant name":
        "A constant name is empty, too long or not NUL-terminated.",
    "constant overlaps command program":
        "A constant region overlaps a command program.",
    # --- strided.py ----------------------------------------------------------
    "geometry profile requires Conv with optional Relu":
        "Only Conv or Conv -> Relu lowers on the geometry profile.",
    "static NCHW tensors required":
        "Input and output shapes must be static NCHW.",
    "constant 2D kernel required":
        "The Conv kernel must be a constant two-element shape.",
    "strides must be 1..4 per axis":
        "Both stride axes must be 1-4.",
    "padding must be 0..K-1 on each side":
        "Each padding value must be 0..K-1.",
    "incorrect convolution output shape":
        "The declared output shape disagrees with kernel, stride and padding.",
    # --- tiled_chain.py ------------------------------------------------------
    "tiles must divide the 8-row chain height (1, 2, 4 or 8)":
        "Choose a tile count that divides the 8-row chain height.",
    "tiled chain supports the chain family's 1x1 and 3x3 kernels":
        "Only the chain family's K1 and K3 kernels tile.",
    "tiled chain requires the 8x8 C3 input":
        "The tiled chain input is fixed at 8x8 C3.",
    "tiled chain supports up to 16 channels per layer":
        "A tiled chain layer may hold at most 16 channels.",
    "tiled chain would need %d tasks; the loader table holds 64":
        "Reduce tiles or layers to fit the loader's 64-task table.",
    # --- transposed.py -------------------------------------------------------
    "dense ConvTranspose requires Conv[/Relu] stem with static 8x8 C1..16 output":
        "The stem must be Conv[/Relu] with a static 8x8 C1-16 output.",
    "dense ConvTranspose supports square K3 or K5":
        "Only square K3 or K5 dense ConvTranspose lowers.",
    "dense ConvTranspose requires per-axis stride1/2 and legal output_padding":
        "Per-axis stride must be 1-2 with output_padding below the stride.",
    "dense ConvTranspose output_shape requires two dimensions and no explicit pads":
        "output_shape takes two values and excludes explicit pads.",
    "dense ConvTranspose output_shape outside bounded profile":
        "The requested output_shape is outside the bounded profile.",
    "unsupported dense ConvTranspose auto_pad":
        "auto_pad must be NOTSET, VALID, SAME_UPPER or SAME_LOWER.",
    "dense ConvTranspose supports C1..16 to C1..16, K3/K5 and per-axis stride1/2":
        "Dense ConvTranspose bounds are C1-16 to C1-16, K3/K5 and per-axis stride 1-2.",
    "grouped ConvTranspose dense rewrite supports input/output C1..16":
        "The grouped dense rewrite needs input and output C1-16.",
    "constant K3 dilation2 weights required":
        "The K3 dilation-2 rewrite needs a constant weight tensor.",
    "K3 dilation2 ConvTranspose requires constant float32 (C_in,C_out/group,3,3) weights":
        "Weights must be constant float32 with that exact shape.",
    "ConvTranspose dilation is supported only for square K2 or depthwise K3 with dilation2":
        "Only square K2 or depthwise K3 with dilation 2 lowers.",
    "rectangular ConvTranspose rewrite requires per-axis padding below its kernel":
        "The rectangular rewrite needs per-axis padding below the axis kernel.",
    "constant depthwise rectangular weights required":
        "The rectangular rewrite needs constant depthwise weights of the declared shape.",
    "unsupported ConvTranspose auto_pad":
        "auto_pad must be NOTSET, VALID, SAME_UPPER or SAME_LOWER.",
    "ConvTranspose output_shape requires two dimensions and no explicit pads":
        "output_shape takes two values and excludes explicit pads.",
    "ConvTranspose output_shape outside bounded profile":
        "The requested output_shape is outside the bounded profile.",
    "ConvTranspose K2/K3/K5 stride1/2 requires padding below K and output_padding below its axis stride":
        "Keep padding below K and output_padding below the axis stride.",
    "ConvTranspose profile requires depthwise C1..16, K2/K3/K5, per-axis stride1/2 and matching output geometry":
        "The geometry is outside the supported depthwise ConvTranspose profile.",
    "constant depthwise square weights required":
        "Depthwise ConvTranspose needs constant square weights [C,1,K,K].",
    "ConvTranspose bias must be float32 and match channels":
        "Bias must be float32 with one value per channel.",
    "ConvTranspose requires 8x8 RGB input":
        "The ConvTranspose stem input is fixed at 8x8 RGB.",
    # --- walk.py -------------------------------------------------------------
    "walk Conv requires default domain with constant weights and bias":
        "Walk Convs need the default domain and constant weights and bias.",
    "walk Conv supports square K1/K3 kernels":
        "Only square K1 or K3 Convs walk.",
    "unsupported walk Conv attributes":
        "A walk Conv carries an attribute outside the allowed set.",
    "walk pool requires the default domain":
        "Walk pools must use the default domain.",
    "walk pool supports 2x2 stride-2 MaxPool/AveragePool only":
        "Only 2x2 stride-2 MaxPool or AveragePool walks.",
    "unsupported chain walk graph":
        "The graph is outside the supported chain-walk profile.",
    "walk chain must start with a Conv":
        "A chain walk must begin with a Conv.",
    "walk native input supports up to 16 channels":
        "The native walk input supports at most 16 channels.",
    "walk native input geometry requires height tiling":
        "The native walk geometry must be height-tiled; check the atom budget.",
    "walk elementwise stage band disagrees with the emitter":
        "The constant elementwise stage recomputed a band other than the one recorded while its "
        "feeding Conv was quantized; the stage's operand scales and the Conv bookkeeping diverged.",
    "unsupported join walk graph":
        "The graph is outside the supported join-walk profile.",
    "the join walk output override requires a Mul join or a Conv tail":
        "Only a Mul join or a Conv tail can carry the join-walk output override.",
    # --- dynamic expressions with no literal text ----------------------------
    "(dynamic) f'{path}: expected uint8/float32 [N,{','.join(map(str, shape[1:]))}]'":
        "A calibration .npy sample has the wrong dtype or shape; export it as uint8/float32 "
        "with the graph's NCHW shape.",
    "(dynamic) f'{path}: calibration inputs must be finite values in [0,255]'":
        "A calibration sample holds NaN/inf or values outside [0,255]; fix the named sample.",
    "(dynamic) spec['error']":
        "A chain-shaped graph whose constant elementwise stage is outside the chain-walk "
        "envelope; the suffix names the stage and the bound it broke.",
    "(dynamic) _elementwise_error(op['index'], op['op'], 'Mul output scale is outside the verified conversion range')":
        "The Mul operand scales fold into an output scale the verified conversion cannot "
        "represent; rescale the feeding Conv or fold the constant into its weights.",
    "open-rknpu: %s\n":
        "The CLI top-level error wrapper; the formatted suffix is the compiler message it caught.",
    # --- mutable.py (the v4 constant-region API) -----------------------------
    "v5 containers have no constant descriptor table; mutable parameters require a v4 container":
        "Only a v4 container carries named constant regions. Compile with mutable_weights=True "
        "or mutable_constants=True (open_rknpu.mutable.compile_mutable).",
    "legacy containers have no constant descriptor table; mutable parameters require a v4 container":
        "Legacy ORNPUBIN containers have no constant table; mutable parameters need an ORNPUSEQ v4 "
        "container from the native Conv or constant-Mul profile.",
    "no constant region named ":
        "The name does not match any descriptor in this container; list them with "
        "open_rknpu.mutable.constant_regions(binary) and pass the exact name.",
    "constant region index ":
        "The index is outside the container's constant table; the message names the table length.",
    "replacement has ":
        "A region is replaced whole: the replacement must be exactly the descriptor's size. Read "
        "the current bytes with constant_payload(binary) and repack for the same band.",
    "the donor has no constant region named ":
        "The donor container must carry the same region name; compile it with the same mutable "
        "flag and the same profile.",
    "the containers do not share a task program, so their bands differ (first difference at byte ":
        "The band's multiplier/shift/zero-point registers live in the task program, so a region "
        "from another band would compute wrong numbers. Pin output_range when compiling the donor, "
        "then compare program_bytes before grafting.",
    "compile_mutable needs mutable_weights or mutable_constants":
        "Ask for a replaceable region: compile_mutable(model, mutable_weights=True) for the native "
        "Conv parameters, or mutable_constants=True for the constant-Mul factor.",
    "the compiled profile has no mutable constant region; no v4 container was emitted":
        "The profile that accepted the graph has no mutable form (only the native Conv and the "
        "constant-Mul profiles emit v4 constants); see docs/api-stability.md.",
    # --- the F1/F2 front-end lowerings ---------------------------------------
    "1-D rank promotion requires rank-3 outputs; '%s' is rank %d":
        "A rank-3 graph must keep rank-3 outputs: the promotion to [N,C,1,L] cannot rewrite a "
        "graph whose output is rank 2 (or 4). Add or remove the reshaping node yourself.",
    "1-D rank promotion cannot rewrite %s node '%s'":
        "ONNX 1-D models are promoted to the 2-D form by rewriting Conv/pool attributes in place; "
        "another node type in the graph needs an explicit reshape before compiling.",
    "1-D rank promotion requires constant rank-3 weights for Conv node '%s'":
        "A 1-D Conv needs constant [O,I,K] weights; dynamic or pre-reshaped weights cannot be "
        "promoted to [O,I,1,K].",
    # --- the F13 profile references (guards the reference does not model) -----
    "pool reference supports MaxPool or AveragePool":
        "The pooling reference models the two verified kinds only; the container came from a "
        "different path, so replay it with the suite's own recorded expected bytes.",
    "pool reference requires at least one 2x2 pooling level":
        "pool_reference models 1-3 chained 2x2/stride-2 pools; a different geometry has no "
        "reference formula here.",
    "reduction reference models exactly three 2x2 pooling levels":
        "The reduction profile (legacy 5/6) is exactly three 2x2 pools; use pool_reference for "
        "one or two levels.",
    "network reference supports MaxPool or AveragePool":
        "The legacy 7/8 network reference models the two verified pool kinds only.",
    "network reference models exactly three 2x2 pooling levels":
        "The legacy 7/8 profile is Conv-Relu-Conv plus exactly three 2x2 pools.",
    "transposed reference requires four pad values":
        "ConvTranspose pads are [beginH, beginW, endH, endW]; pass the container's own values.",
    "transposed reference supports per-axis stride 1 or 2":
        "The transposed reference models stride 1 or 2 per axis; other strides have no verified "
        "container to compare against.",
    "transposed reference requires output_padding below its axis stride":
        "ONNX requires output_padding < stride on each axis; the container was emitted with a "
        "legal value, so this means the arguments were mixed up.",
    "transposed reference supports K2/K3/K5 kernels":
        "The transposed emitter rewrites K2/K3/K5 (and small rectangular kernels) into the "
        "verified forms; the reference does not model other kernel sizes.",
    "transposed reference requires a square kernel weight layout":
        "That rewrite path packs a square kernel; a rectangular layout belongs to the "
        "rectangular rewrite the emitter reports instead.",
    "transposed reference requires the stem channels to match the weight layout":
        "The stem channel count must agree with the weight tensor's input channels; recompile "
        "and read the layout from the container.",
    "transposed reference output_shape must match the emitted channels":
        "The output_shape argument names a channel count the weights do not produce; omit it to "
        "derive the geometry from the container.",
    # --- calibration parity (F3) --------------------------------------------
    "calibration is unsupported for the height-strip tiled chain":
        "The height-strip tiled chain cannot carry per-stage measured bands yet; compile the "
        "same graph with the untiled chain profile when you need calibration_ranges.",
}


def write_documents(out=DOCS):
    """Write both generated documents into ``out`` (created if needed)."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "registers.md").write_text(build_registers_doc(), encoding="utf-8")
    (out / "errors.md").write_text(build_errors_doc(), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DOCS), help="directory for the generated documents")
    args = parser.parse_args(argv)
    write_documents(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



