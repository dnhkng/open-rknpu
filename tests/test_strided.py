"""MIT. Public stride-2 lowering and profile boundaries."""
from pathlib import Path
import tempfile
import unittest
import onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
ROOT=Path(__file__).resolve().parents[1]/'research/stride2_suite'
class StridedTests(unittest.TestCase):
    def test_verified_programs(self):
        for i in range(3):
            with tempfile.TemporaryDirectory() as tmp:
                m=onnx.load(ROOT/f'model{i:03}.onnx')
                m.graph.node[0].attribute.append(h.make_attribute('strides',[2,2]))
                for d in m.graph.output[0].type.tensor_type.shape.dim[2:]:d.dim_value=4
                p=Path(tmp)/'model.onnx';onnx.save(m,p)
                data,meta=compile_sequence(p)
                self.assertEqual(data,(ROOT/f'model{i:03}.bin').read_bytes())
                self.assertEqual(decode_sequence(data)['output_bytes'],48)
                m.graph.node[0].attribute[-1].ints[:]=[3,3];onnx.save(m,p)
                with self.assertRaises(ValueError):compile_sequence(p)
