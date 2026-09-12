"""Two-layer compiler boundaries and preserved independent hardware artifacts."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper,numpy_helper
from open_rknpu.chain import chain_reference, native_reference
from open_rknpu.compiler import compile_model
from open_rknpu.quantization import Quantization, reference  # noqa: F401 (used by the reference tests)

ROOT=Path(__file__).resolve().parents[1]/"research/generated"

class ChainTests(unittest.TestCase):
    def test_payloads_match_independently_verified_hardware_programs(self):
        for channels in (3,4,8,16):
            with self.subTest(channels=channels):
                path=ROOT/f"chain_heldout{channels}.onnx"
                payload,meta=compile_model(path)
                self.assertEqual(payload,path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["profile"],2)

    def test_reject_unsupported_second_layer(self):
        for kind in ("stride","bias","nonfinite","dynamic"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                model=onnx.load(ROOT/"chain_heldout4.onnx")
                if kind=="stride":
                    model.graph.node[2].attribute.append(helper.make_attribute("strides",[2,2]))
                elif kind in ("bias","nonfinite"):
                    name="b2" if kind=="bias" else "w2"
                    t=next(t for t in model.graph.initializer if t.name==name)
                    a=np.zeros((1,),np.float32) if kind=="bias" else np.full((3,4,1,1),np.nan,np.float32)
                    t.CopyFrom(numpy_helper.from_array(a,name))
                else:
                    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param="height"
                p=Path(tmp)/"bad.onnx";onnx.save(model,p)
                with self.assertRaises(ValueError):compile_model(p)

    def test_reject_unsupported_output_override(self):
        with self.assertRaisesRegex(ValueError,"overrides"):
            compile_model(ROOT/"chain_heldout4.onnx",1.0,0)


class ChainReferenceTests(unittest.TestCase):
    """`chain_reference` pads the second layer with the first layer's zero point (F4).

    The emitter programs register 0x1184 with the band the second Conv reads; the reference
    must model exactly that. Before the fix both used the register's -128 reset, so the two
    agree only when the first layer's band *is* zero point -128.
    """

    @classmethod
    def setUpClass(cls):
        import dataclasses
        from open_rknpu.scheduler import compile_sequence
        _, meta = compile_sequence(ROOT / "chain_heldout4.onnx")
        cls.dataclasses = dataclasses
        cls.first = Quantization(**{k: np.array(v) if isinstance(v, list) else v
                                    for k, v in meta["first"]["quantization"].items()})
        cls.second = Quantization(**{k: np.array(v) if isinstance(v, list) else v
                                     for k, v in meta["second"].items()})
        cls.case = np.random.default_rng(3).integers(0, 256, (8, 8, 3), dtype=np.uint8)

    def test_the_border_is_the_first_layers_zero_point(self):
        first = self.dataclasses.replace(self.first, output_zero_point=-17)
        propagated = chain_reference(self.case, first, self.second)
        self.assertTrue(np.array_equal(propagated,
                                       native_reference(reference(self.case, first), self.second, -17)))
        # Whether a particular sample *exposes* the border difference is data dependent (a
        # sample whose border taps contribute nothing looks identical either way), so the
        # convention itself is pinned at the register level by the emit-semantics and tiled
        # chain tests rather than by inequality here.

    def test_a_zero_point_reset_first_layer_matches_the_reset(self):
        self.assertEqual(self.first.output_zero_point, -128)
        self.assertTrue(np.array_equal(chain_reference(self.case, self.first, self.second),
                                       native_reference(reference(self.case, self.first), self.second)))


if __name__ == "__main__":
    unittest.main()
