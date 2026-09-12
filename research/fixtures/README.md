# Vendor-oracle fixtures

The `model.rknn` files and decoded dumps in these directories are **comparison evidence**,
not part of the distribution:

* the `.rknn` files were produced by the vendor RKNN toolkit from ONNX graphs that this
  repository generates (`research/build_*_oracle.py`), so the model topology is ours;
* the numbered dumps (`model.rknn.0`, `.1`, ...), `.bfd` listings and `.replay` traces are
  the vendor container's sections and task programs, kept so the vendor's choices can be
  compared against our emitters;
* nothing here is required to build, test or run open-rknpu. The open emitters never read
  these files, and `tests/test_composer_emitters.py` compares the *compiler's* output
  against the retained board evidence instead.

Provenance of the vendor toolkit itself, the toolchain and the datasets is in
[THIRD_PARTY.md](../../THIRD_PARTY.md) and [docs/provenance.md](../../docs/provenance.md).
