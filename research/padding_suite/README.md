# Leading-Pad host preprocessing (constant / reflect / edge / wrap)

The CNA border path injects only the activation zero point, so constant padding is
native; reflection, edge/replication and circular borders have no verified NPU
producer on this profile. `compile --sequence` folds a leading `Pad` node on the
graph input into the input shape and records the required preprocessing:

```json
"input_padding": {"mode": "edge", "pads": [1,1,1,1],
                  "padded_shape": [1,3,10,10], "requires_host_preprocessing": true}
```

The caller pads the packed input with `open_rknpu.padding.pad_input(array, pads, mode)`
before `ornpu_run`, which keeps all tensor math on the NPU and needs no register
program for a gather the hardware cannot express.

**Verified on RV1103: 12 models, 192 inferences, 66,048 exact output bytes**
(`board_api_test padding_suite 12`), covering all four modes with pads
`(1,1,1,1)`, `(2,2,2,2)` and the asymmetric `(0,1,2,1)`. Expected bytes come from
the native input reference over the padded input.

## Reproduction

```sh
PYTHONPATH=src python research/build_padding_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test padding_suite 12
```

`tests/test_padding.py` checks `pad_input` against NumPy for all modes, the folded
metadata, and rejection of negative pads. A dedicated NPU copy/pad producer for
non-constant borders remains future work in
[completion-plan](../../docs/plans/completion-plan.md) phase P5.
