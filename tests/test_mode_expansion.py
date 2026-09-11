"""MIT. Semantic rewrites and verified-profile boundary checks."""
from pathlib import Path
import tempfile
import unittest
import numpy as np,onnx
from onnx import helper as h
from onnx.reference import ReferenceEvaluator
from open_rknpu.normalize import normalize_model
from open_rknpu.scheduler import compile_sequence
ROOT=Path(__file__).resolve().parents[1]/'research'
class ModeExpansionTests(unittest.TestCase):
    def test_lowering_float_semantics_and_immutability(self):
        rng=np.random.default_rng(77)
        for suite in ('lowered_conv_suite','constant_mul_suite'):
            for p in sorted((ROOT/suite).glob('model*.onnx')):
                m=onnx.load(p);saved=m.SerializeToString();n=normalize_model(m)
                self.assertEqual(m.SerializeToString(),saved)
                x=rng.uniform(-5,5,(1,3,8,8)).astype(np.float32)
                np.testing.assert_allclose(ReferenceEvaluator(m).run(None,{'input':x})[0],ReferenceEvaluator(n).run(None,{'input':x})[0],rtol=1e-5,atol=1e-5)
    def test_verified_depthwise_programs(self):
        for suite,index in [('depthwise_c4_suite',0),('depthwise_expansion_suite',5),('depthwise_expansion_suite',6)]:
            p=ROOT/suite/f'model{index:03}.onnx'
            data,_=compile_sequence(p);self.assertEqual(data,p.with_suffix('.bin').read_bytes())
    def test_verified_depthwise_channel_expansion(self):
        for channels in range(5,17):
            with self.subTest(channels=channels):
                p=ROOT/f'depthwise_c{channels}_suite/model000.onnx'
                data,_=compile_sequence(p)
                self.assertEqual(data,p.with_suffix('.bin').read_bytes())

    def test_spatial_reshape_and_channel_boundary(self):
        p=ROOT/'spatial_reshape_suite/model000.onnx'
        data,_=compile_sequence(p);self.assertEqual(data,p.with_suffix('.bin').read_bytes())
        m=onnx.load(p)
        # An overridable shape must never become a static metadata-only operation.
        m.graph.input.append(h.make_tensor_value_info('shape',onnx.TensorProto.INT64,[4]))
        with tempfile.TemporaryDirectory() as tmp:
            q=Path(tmp)/'bad.onnx';onnx.save(m,q)
            with self.assertRaises(ValueError):compile_sequence(q)
    def test_mul_public_conv_output_is_preserved(self):
        m=onnx.load(ROOT/'constant_mul_suite/model000.onnx')
        m.graph.output.append(h.make_tensor_value_info('conv',1,[1,3,8,8]))
        n=normalize_model(m)
        self.assertEqual([v.op_type for v in n.graph.node],['Conv','Mul'])
