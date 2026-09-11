"""Two-layer compiler boundaries and preserved independent hardware artifacts."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper,numpy_helper
from open_rknpu.compiler import compile_model

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
