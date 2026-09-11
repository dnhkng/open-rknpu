"""MIT. Public depthwise stride2 matches independently tested programs."""
from pathlib import Path
import json
import tempfile
import unittest
import onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
ROOT=Path(__file__).resolve().parents[1]/'research'
class DepthwiseStrideTests(unittest.TestCase):
    def test_all_verified_models(self):
        manifest=json.loads((ROOT/'depthwise_suite/manifest.json').read_text())
        for i,entry in enumerate(manifest):
            with self.subTest(index=i),tempfile.TemporaryDirectory() as tmp:
                m=onnx.load(ROOT/'depthwise_suite'/f'model{i:03}.onnx')
                m.graph.node[-1].attribute.append(h.make_attribute('strides',[2,2]))
                for d in m.graph.output[0].type.tensor_type.shape.dim[2:]:d.dim_value=4
                p=Path(tmp)/'model.onnx';onnx.save(m,p)
                data,_=compile_sequence(p,entry['input_scale'],entry['input_zero_point'])
                self.assertEqual(data,(ROOT/'depthwise_stride2_suite'/f'model{i:03}.bin').read_bytes())
