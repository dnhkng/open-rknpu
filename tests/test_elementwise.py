"""MIT. Add graph boundaries, rounding and sequence descriptor validation."""
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.elementwise import add_reference, mul_reference, sub_reference, max_reference
from open_rknpu.sequence import decode_sequence
from open_rknpu.model import checksum

ROOT=Path(__file__).resolve().parents[1]/'research/add_suite'

class AddTests(unittest.TestCase):
    def test_signed_halfway_rounding(self):
        a=np.array([-128,-3,-1,1,3,127],np.int8)
        b=np.array([-128,0,0,0,0,127],np.int8)
        np.testing.assert_array_equal(add_reference(a,b),[-128,-2,0,0,2,127])

    def test_shared_branch_quantization_and_descriptor(self):
        data,meta=compile_sequence(ROOT/'model000.onnx',.5,128)
        a,b=meta['branches']
        self.assertEqual(a['output_scale'],b['output_scale'])
        self.assertEqual(a['output_zero_point'],0)
        self.assertEqual(b['output_zero_point'],0)
        info=decode_sequence(data)
        self.assertEqual(info['output_scale'],2*a['output_scale'])
        self.assertEqual(info['tasks'][-1]['enable'],24)
        for enable,mask,count in [(24,3072,78),(24,768,79),(8,768,78)]:
            broken=bytearray(data)
            struct.pack_into('<III',broken,96+2*16+4,count,enable,mask)
            broken[80:84]=b'\0'*4
            struct.pack_into('<I',broken,80,checksum(broken))
            with self.assertRaisesRegex(ValueError,'task descriptor'):decode_sequence(broken)

    def test_reject_unsupported_graphs(self):
        for kind in ['broadcast','extra','connection','shape','weights','stride']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                m=onnx.load(ROOT/'model000.onnx')
                if kind=='broadcast':
                    m.graph.node[-1].input[1]='bb'
                elif kind=='extra':
                    m.graph.node[-1].output[0]='sum';m.graph.node.append(h.make_node('Relu',['sum'],['output']))
                elif kind=='connection':m.graph.node[1].input[0]='a'
                elif kind=='shape':m.graph.input[0].type.tensor_type.shape.dim[2].dim_value=7
                elif kind=='weights':m.graph.input.append(h.make_tensor_value_info('aw',1,[3,3,1,1]))
                elif kind=='stride':m.graph.node[1].attribute.append(h.make_attribute('strides',[2,2]))
                path=Path(tmp)/'bad.onnx';onnx.save(m,path)
                with self.assertRaises((ValueError,onnx.checker.ValidationError,onnx.shape_inference.InferenceError)):
                    compile_sequence(path)


class MulTests(unittest.TestCase):
    def test_signed_halfway_and_saturation(self):
        a=np.array([1,3,-1,-3,-128,127,-128],np.int8)
        b=np.array([64,64,64,64,-128,127,127],np.int8)
        np.testing.assert_array_equal(mul_reference(a,b),[0,2,0,-2,127,126,-127])

    def test_output_quantization(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=onnx.load(ROOT/'model000.onnx')
            model.graph.node[-1].op_type='Mul'
            path=Path(tmp)/'mul.onnx';onnx.save(model,path)
            data,meta=compile_sequence(path,.25,128)
        a,b=meta['branches']
        self.assertGreater(a['output_scale'],0)
        self.assertGreater(b['output_scale'],0)
        self.assertEqual((a['output_zero_point'],b['output_zero_point']),(0,0))
        info=decode_sequence(data)
        self.assertEqual(info['output_scale'],float(np.float32(128*a['output_scale']*b['output_scale'])))
        self.assertEqual(info['output_zero_point'],0)
        self.assertEqual(meta['elementwise_profile'],'mul-8x8-c3-independent-scales')


class SubMaxTests(unittest.TestCase):
    def test_signed_edges(self):
        a=np.array([-128,127,-3,3],np.int8)
        b=np.array([127,-128,0,0],np.int8)
        np.testing.assert_array_equal(sub_reference(a,b),[-128,127,-2,2])
        np.testing.assert_array_equal(max_reference(a,b),[64,64,0,2])

    def test_public_profiles(self):
        for op in ('Sub','Max'):
            with self.subTest(op=op),tempfile.TemporaryDirectory() as tmp:
                m=onnx.load(ROOT/'model000.onnx');m.graph.node[-1].op_type=op
                path=Path(tmp)/'model.onnx';onnx.save(m,path)
                data,meta=compile_sequence(path)
                self.assertEqual(meta['elementwise_profile'],op.lower()+'-8x8-c3-equal-scale')
                self.assertEqual(decode_sequence(data)['output_scale'],2*meta['branches'][0]['output_scale'])
