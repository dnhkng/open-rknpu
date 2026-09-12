# Security Policy

`open-rknpu` is an experimental compiler and runtime. It parses untrusted *shape*
metadata from ONNX models and loads compiler-produced container files into a device
driver, so malformed input is treated as a first-class security concern rather than a
crash to shrug off.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes — the current release line |
| < 0.1 | No |

Only the latest 0.1.x release receives security fixes. Because the container format is
versioned (`ORNPUBIN` v1/v2 and `ORNPUSEQ` v3/v4/v5), a fix that changes on-disk layout
will be called out in [CHANGELOG.md](CHANGELOG.md) and in
[runtime/sequence_format.md](runtime/sequence_format.md).

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report privately through a
[GitHub private security advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on the repository (`Security` → `Report a vulnerability`). That channel keeps the report
visible only to the maintainers until a fix is available.

A useful report includes:

* the affected version or commit,
* the smallest input that triggers the problem (the `.bin` container or `.onnx` model,
  or a generating script if the file cannot be shared),
* the command that was run, and
* what happened versus what was expected — a Python traceback, a `SIGSEGV`, an
  out-of-bounds read reported by a sanitizer, or a corrupted output.

If the issue is exploitable, describe the impact you believe it has; you do not need to
produce a working proof of concept, but one shortens the fix considerably.

## Scope

In scope:

* **The compiler** (`src/open_rknpu/`) — the ONNX front end, normalizer, scheduler,
  composer, emitters and container writer. Malformed graphs that cause an unhandled
  exception, an infinite loop, unbounded memory growth or invalid emitted bytes are in
  scope, as are any cases where a bound is checked incorrectly.
* **The container parser/encoder** (`open_rknpu.sequence` and the legacy decoder) — the
  Python-side reader, validator and writer.
* **The C loader and runtime** (`runtime/open_rknpu.c`, `runtime/main.c`) — especially
  **malformed containers**. The loader validates the magic, the format version, the
  header size, descriptor bounds, task and tensor counts, tensor lengths and offsets,
  arena and payload sizes and their alignment, the declared geometry and quantization
  fields, and the FNV-1a checksum of the complete file with the checksum field zeroed
  before any of that data reaches the driver. A way to bypass, mis-order or overflow any
  of those checks — a length that wraps, an offset that escapes the arena, a count that
  overruns a table, a checksum that is accepted when it should not be — is a security
  bug and the most valuable class of report here.

Out of scope:

* **Physical board damage.** Running a container on real hardware is at your own risk;
  a bug that requires you to flash, miswire or over-drive a board is not a security
  vulnerability of this project.
* **Vendor toolchain issues.** The Rockchip cross toolchain, `librknn*`, RKNN toolkit
  components and anything else distributed by the vendor are not ours to fix — report
  those to their vendor.
* **The vendor kernel driver.** `rknpu` / `/dev/rknpu`, including the kernel module,
  its ioctl surface and its memory management, belongs to the vendor's kernel tree.
  A driver flaw that our loader merely triggers is a driver issue; we will still
  document a workaround if one exists on our side.
* Missing hardening that requires privileged local access to the board anyway.

## Response expectations

This is a volunteer, research-oriented project, so these are targets rather than a
contract:

* **Acknowledgement** of a private report within **7 days**.
* **Initial assessment** (in scope / not, severity, whether a fix is planned) within
  **14 days**.
* **Fix or mitigation** for a confirmed issue in the compiler, parser or loader within
  **90 days** of the acknowledgement, released as a patch version where possible.
* **Coordinated disclosure**: we will agree a disclosure date with you, credit you in
  the advisory and the changelog unless you prefer otherwise, and publish a GitHub
  security advisory once a fixed release is available.

If a report is going to take longer than the windows above, we will say so with a
revised estimate rather than leave it open silently.

## Verification you can do yourself

* The container loader is the highest-risk surface. Its checks are documented in
  [docs/container-format.md](docs/container-format.md) and
  [runtime/sequence_format.md](runtime/sequence_format.md); the byte layout is public
  so you can construct adversarial inputs directly.
* If the repository contains `tests/test_container_fuzz.py`, that is the home of the
  malformed/truncated/mutated-container fuzzing corpus for the loader — run it with
  `PYTHONPATH=src python -m unittest tests.test_container_fuzz` and extend it with any
  input that reproduces a finding.
* The host test suite and the container baseline are the regression net for a fix:
  `make test`, `make coverage` and `PYTHONPATH=src python research/verify_suites.py`.
