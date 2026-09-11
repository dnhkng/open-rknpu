"""MIT. Graph safety boundaries and per-channel depthwise arithmetic."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.chain import native_quantize,native_reference
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.quantization import Quantization

ROOT=Path(__file__).resolve().parents[1]/'research/depthwise_suite'

class DepthwiseTests(unittest.TestCase):
    def test_grouped_reference_matches_diagonal_dense(self):
        rng=np.random.default_rng(551)
        w=rng.uniform(-.5,.5,(3,1,3,3)).astype(np.float32)
        q=native_quantize(w,np.array([1,-1,.5],np.float32),.25,-37,symmetric=True)
        dense=np.zeros((3,3,3,3),np.int64)
        for c in range(3):dense[c,c]=q.weights[c].reshape(3,3)
        d=Quantization(**dict(vars(q),weights=dense.reshape(3,-1)))
        x=rng.integers(-128,128,(8,8,3),dtype=np.int8)
        np.testing.assert_array_equal(depthwise_reference(x,q,-37),native_reference(x,d,-37))

    def test_symmetric_positive_negative_and_zero_channels(self):
        w=np.array([np.ones((1,3,3)), -np.ones((1,3,3)),np.zeros((1,3,3))],np.float32)
        q=native_quantize(w,np.array([0,0,1],np.float32),.5,11,symmetric=True)
        np.testing.assert_array_equal(q.weight_zero_points,0)
        self.assertTrue(np.isfinite(q.weight_scales).all())
        np.testing.assert_array_equal(q.weights[0],127)
        np.testing.assert_array_equal(q.weights[1],-127)
        np.testing.assert_array_equal(q.weights[2],0)

    def test_reject_unsupported_graphs(self):
        for kind in ['group','stride','dilation','pads','shape','weights','branch','extra']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                m=onnx.load(ROOT/'model000.onnx');node=m.graph.node[-1]
                if kind in ('group','stride','dilation','pads'):
                    name={'stride':'strides','dilation':'dilations'}.get(kind,kind)
                    kept=[a for a in node.attribute if a.name!=name];del node.attribute[:];node.attribute.extend(kept)
                    node.attribute.append(h.make_attribute(name,1 if kind=='group' else [0]*4 if kind=='pads' else [2,2]))
                elif kind=='shape':m.graph.input[0].type.tensor_type.shape.dim[2].dim_value=9
                elif kind=='weights':
                    m.graph.input.append(h.make_tensor_value_info('w2',1,[3,1,3,3]))
                elif kind=='branch':node.input[0]='input'
                elif kind=='extra':
                    node.output[0]='dw';m.graph.node.append(h.make_node('Relu',['dw'],['output']))
                path=Path(tmp)/'bad.onnx';onnx.save(m,path)
                with self.assertRaises((ValueError,onnx.checker.ValidationError)):
                    compile_sequence(path)

    def test_sequence_exposes_final_quantization(self):
        from open_rknpu.sequence import decode_sequence
        binary,meta=compile_sequence(ROOT/'model000.onnx',.25,128)
        info=decode_sequence(binary)
        self.assertEqual(info['output_shape_nhwc'],[1,8,8,3])
        self.assertTrue(info['serial'])
        self.assertEqual(info['input_zero_point'],128)
        self.assertEqual(info['output_scale'],meta['depthwise']['output_scale'])
