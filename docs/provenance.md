# Provenance and clean-room statement

This page states exactly where the knowledge in this repository came from, because the
project's central claim is that it is an **independent, open implementation** and not a
wrapper, a decompilation dump or a repackaging of vendor code.

## What the compiler and runtime contain

Everything in `src/open_rknpu/` and `runtime/` was written for this project. The compiler
emits register words, weight/quantization blocks and container bytes from its own tables
(`src/open_rknpu/register_profile.py`, `runtime/sequence_format.md`); the runtime uses
`open`, `mmap`, `ioctl` and `munmap` against `/dev/rknpu` and libc, nothing else. No vendor
header, object file, library or decompiled source is included, linked or read at build or
run time.

## Where the register knowledge came from

1. **Board experiments (the bulk).** Every register value, task enable mask, tail-control
   rule, quantization convention and format field recorded here was measured on the
   attached Luckfox Pico Mini B by submitting containers and reading outputs, and is
   reproduced by the suites under `research/` with their `board_results_*.json`. The
   chronological record, including the hypotheses that failed, is
   [investigation-log.md](investigation-log.md).
2. **Public documentation and public kernel sources.** Rockchip's kernel driver for this NPU
   is public (GPL-2.0). It was read as documentation for ioctl numbers, structure layouts,
   job-flag semantics and which driver actions are implemented. Those upstream files are
   **not redistributed**; the provenance (upstream repository, commit, per-file sha256) is
   in [research/vendor/README.md](../research/vendor/README.md) and
   [research/vendor/provenance.json](../research/vendor/provenance.json). Register values
   from that source are facts about the interface, re-expressed in this project's own
   tables and documentation.
3. **The vendor runtime as a black-box oracle.** During development the vendor RKNN toolkit
   compiled the same ONNX graphs so that their emitted task programs could be compared with
   ours. It was used only in scripts named `*_oracle.py` and only as a comparison target:
   the open emitters never read a vendor artifact, and no vendor model, binary or capture
   is required to build, test or run this project. The `*.rknn` fixtures and decoded
   captures under `research/fixtures/` are outputs of that comparison for **our own**
   models, kept as evidence of what the vendor did.
4. **Related-hardware documents.** The RK3588 TRM and vendor register descriptions were
   consulted for background and are deliberately **not** redistributed.

No vendor binary was disassembled to produce compiler code. Where an entry in the
investigation log refers to a decompiled listing, it is recorded as a *hypothesis* that was
then confirmed or refuted on the board.

## Why this matters for you

* You do not need any Rockchip SDK, RKNN package or vendor library to install, build or run
  this project. The only external pieces are NumPy and ONNX on the host, and a C compiler
  for the board runtime.
* You do need a toolchain to build the board runtime. The repository does not ship one;
  `research/fetch_toolchain.py` downloads the Luckfox toolchain for convenience, and any
  ARM uClibc cross compiler with the board's sysroot works.
* If you are evaluating licence compatibility, read [THIRD_PARTY.md](../THIRD_PARTY.md):
  the shipped code is MIT, one dual-licensed header is used under its MIT option, and the
  GPL-2.0 kernel sources are referenced but not redistributed.

## Release artifacts

The knowledge above is about *inputs*; the release itself carries its own provenance:

* **Build provenance attestation.** `.github/workflows/release.yml` attests the `sdist` and
  `wheel` it builds with GitHub's `attest-build-provenance` action (OIDC, no stored secret),
  so anyone can verify which workflow, commit and repository produced a downloaded file:

  ```sh
  gh attestation verify dist/open_rknpu-*.whl --repo dnhkng/open-rknpu
  ```

* **Signed tags (maintainer step).** Release commits are tagged with a GPG/SSH-signed
  annotated tag; the signing key is the maintainer's and deliberately not in CI. Create and
  verify one with:

  ```sh
  git tag -s v0.1.0 -m "open-rknpu v0.1.0"
  git verify-tag v0.1.0
  git push origin v0.1.0
  ```

  GitHub shows the verified badge next to the tag once the public key is uploaded to the
  maintainer's account. `docs/publish-checklist.md` (row A7) tracks this.
* **Reproducible contents.** `research/check_reproducible_build.py` normalises the sdist
  (timestamps, uid/gid, modes, member order) and audits that it carries the compiler and its
  attribution files and not the research evidence tree.

## Corrections

If you find a claim in this repository that cannot be reproduced from the committed
evidence, the honest outcome is a correction and a new suite, not a footnote. Open an issue
with the suite directory, the command and the observed output.
