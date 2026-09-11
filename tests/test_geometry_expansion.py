"""MIT. Expanded geometry and public transposed-kernel regression checks."""
from pathlib import Path
import json,tempfile,unittest
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
ROOT=Path(__file__).resolve().parents[1]/'research'
class GeometryExpansionTests(unittest.TestCase):
    def test_transposed_matches_independent_program(self):
        for i in range(6):
            with self.subTest(i=i),tempfile.TemporaryDirectory() as tmp:
                info=json.loads((ROOT/'transpose_learned_suite'/f'model{i:03}.json').read_text())
                m=onnx.load(ROOT/'depthwise_suite'/f'model{i:03}.onnx');n=m.graph.node[-1]
                n.op_type='ConvTranspose';del n.attribute[:]
                n.attribute.extend([h.make_attribute('group',3),h.make_attribute('kernel_shape',[2,2]),h.make_attribute('strides',[2,2])])
                for t in m.graph.initializer:
                    if t.name==n.input[1]:t.CopyFrom(nh.from_array(np.array(info['weights'],np.float32),t.name))
                    elif t.name==n.input[2]:t.CopyFrom(nh.from_array(np.array(info['bias'],np.float32),t.name))
                for d in m.graph.output[0].type.tensor_type.shape.dim[2:]:d.dim_value=16
                path=Path(tmp)/'model.onnx';onnx.save(m,path)
                data,_=compile_sequence(path)
                self.assertEqual(data,(ROOT/'transpose_learned_suite'/f'model{i:03}.bin').read_bytes())

    def test_transposed_padding_and_output_shape_programs(self):
        for suite in ('transpose_padding_suite','transpose_output_shape_suite','transpose_overlap_suite','transpose_no_bias_suite','transpose_stride1_suite','transpose_unequal_suite','transpose_dilation_suite','transpose_rectangular_suite','transpose_channels_suite','transpose_dense_suite','transpose_grouped_suite'):
            for p in (ROOT/suite).glob('model*.onnx'):
                self.assertEqual(compile_sequence(p)[0],p.with_suffix('.bin').read_bytes())
    def test_verified_geometry_programs(self):
        for suite in ('conv_geometry_suite','conv_boundary_suite','mul_geometry_suite','add_geometry_suite','sub_geometry_suite','max_geometry_suite','two_input_mul_suite','standalone_mul_suite','mul_sources_suite','mul_batch_suite','mul_two_input_batch_suite','mul_reshape_suite','mul_broadcast_suite','mul_broadcast_output_suite','mul_output_quantization_suite','mul_output_rounding_suite','mul_boundary_zero_points_suite','mul_external_intermediate_suite','mul_relu_suite','residual_add_suite','lut_public_suite','leaky_verified_suite','prelu_public_suite','native_input_suite','native_combined_suite','group_dilation_suite','conv_stride34_suite','native_stride34_suite','group_dense_suite','group_expansion_suite','depthwise_rewrite_expansion_suite','weight_edge_suite','output_channels_complete_suite','border_padding_suite','auto_padding_suite','native_mul_suite','native_elementwise_suite','transpose_output_quantization_suite','depthwise_output_quantization_suite','conv_clip_suite'):
            manifest=json.loads((ROOT/suite/'manifest.json').read_text())
            for entry in manifest:
                i=entry['index'];p=ROOT/suite/f'model{i:03}.onnx'
                output_range=(entry.get('compile_output_range') or
                              ({'scale':entry['output_scale'],'zero_point':entry['output_zero_point']}
                               if suite=='mul_output_quantization_suite' else None))
                data,_=compile_sequence(p,entry['input_scale'],entry['input_zero_point'],
                                        output_range=output_range,
                                        mul_operand_zero_points=tuple(entry.get('mul_operand_zero_points',(0,0))))
                self.assertEqual(data,p.with_suffix('.bin').read_bytes())
    def test_depthwise_geometry(self):
        for suite in ('depthwise_geometry_suite','depthwise_no_bias_suite'):
            for p in (ROOT/suite).glob('model*.onnx'):
                data,_=compile_sequence(p);self.assertEqual(data,p.with_suffix('.bin').read_bytes())

    def test_new_composed_programs(self):
        for suite in ('chain_output_quantization_suite','mul_batch_broadcast_suite','depthwise_pointwise_suite','mul_after_relu_suite','mul_relu_order_suite','mul_add_suite','accumulator_boundary_suite'):
            for entry in json.loads((ROOT/suite/'manifest.json').read_text()):
                p=ROOT/suite/f"model{entry['index']:03}.onnx"
                data,_=compile_sequence(p,entry['input_scale'],entry['input_zero_point'],entry['compile_output_range'])
                self.assertEqual(data,p.with_suffix('.bin').read_bytes())
        for entry in json.loads((ROOT/'mul_clip_suite/manifest.json').read_text()):
            p=ROOT/'mul_clip_suite'/f"model{entry['index']:03}.onnx"
            self.assertEqual(compile_sequence(p)[0],p.with_suffix('.bin').read_bytes())

    def test_qlinearconv_import_programs(self):
        for suite in ('qlinearconv_import_suite','qdq_conv_import_suite'):
            for p in (ROOT/suite).glob('model*.onnx'):
                data,meta=compile_sequence(p)
                self.assertEqual(data,p.with_suffix('.bin').read_bytes())
            self.assertIn('preserved bit-for-bit',meta['quantized_import'])
        with self.assertRaisesRegex(ValueError,'carries its own quantization'):
            compile_sequence(ROOT/'qlinearconv_import_suite/model000.onnx',.5,128)

    def test_conv_final_halfway_reference(self):
        from open_rknpu.quantization import Quantization,reference
        entry=json.loads((ROOT/'mul_geometry_suite/manifest.json').read_text())[7]['branches'][1]
        q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in entry['quantization'].items()})
        x=np.fromfile(ROOT/'conv_halfway_suite/input000.u8',np.uint8).reshape(32,5,8,3)
        expected=np.fromfile(ROOT/'conv_halfway_suite/expected_even.i8',np.int8).reshape(32,5,8,16)
        np.testing.assert_array_equal(np.stack([reference(v,q) for v in x]),expected)

    def test_legacy_single_channel_bias_only(self):
        from open_rknpu.compiler import compile_model
        from open_rknpu.model import encode,decode
        payload,meta=compile_model(ROOT/'conv_boundary_suite/model000.onnx')
        self.assertEqual(decode(encode(payload,meta))['output_bytes'],25)

    def test_leaky_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError,'overflow INT32'):
            compile_sequence(ROOT/'leaky_verified_suite/model020.onnx',.5,255)

    def test_broadcast_shape_and_overridable_constant_rejected(self):
        m=onnx.load(ROOT/'mul_broadcast_suite/model001.onnx')
        m.graph.input.append(h.make_tensor_value_info('factor',1,[1,3,1,1]))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.onnx';onnx.save(m,p)
            with self.assertRaises(ValueError):compile_sequence(p)

    def test_rectangular_float_equivalence(self):
        from onnx.reference import ReferenceEvaluator
        from open_rknpu.normalize import normalize_model
        rng=np.random.default_rng(193)
        for p in (ROOT/'rectangular_conv_suite').glob('model*.onnx'):
            m=onnx.load(p);n=normalize_model(m);x=rng.uniform(-2,2,(1,3,8,8)).astype(np.float32)
            np.testing.assert_allclose(ReferenceEvaluator(m).run(None,{'input':x})[0],ReferenceEvaluator(n).run(None,{'input':x})[0],atol=1e-5,rtol=1e-5)

    def test_native_large_integer_outputs(self):
        from open_rknpu.native import compile_native_input,native_input_reference
        from open_rknpu.quantization import Quantization
        for suite in ('native_large_suite','native_large_combined_suite','native_output_blocks_suite','native_input_blocks_suite','native_blocks_combined_suite','native_large_kernels_combined_suite','native_kernel_limits_suite','native_spatial_limits_suite','native_spatial_tiling_suite','native_output_limits_suite','native_batch_suite','native_dilation_verified_suite'):
            for entry in json.loads((ROOT/suite/'manifest.json').read_text()):
                p=ROOT/suite/f"model{entry['index']:03}.onnx"
                data,meta=compile_native_input(onnx.load(p),entry['input_scale'],entry['input_zero_point'])
                self.assertEqual(data,p.with_suffix('.bin').read_bytes())
                q=Quantization(**{k:np.array(v) if isinstance(v,list) else v for k,v in meta['quantization'].items()})
                x=np.fromfile(p.with_name(p.stem.replace('model','input')+'.u8'),np.uint8).reshape(-1,*entry['input_shape'])
                y=np.stack([native_input_reference(v,q,entry['input_zero_point'],meta['conv_pads'],meta['conv_strides'],meta.get('conv_dilations',(1,1))) for v in x])
                self.assertEqual(y.tobytes(),p.with_name(p.stem.replace('model','expected')+'.i8').read_bytes())

    def test_group_and_dilation_rewrite_equivalence(self):
        from onnx.reference import ReferenceEvaluator
        from open_rknpu.normalize import normalize_model
        rng=np.random.default_rng(1931)
        for p in (ROOT/'group_dilation_suite').glob('model*.onnx'):
            m=onnx.load(p);n=normalize_model(m)
            shape=[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
            x=rng.uniform(-2,2,shape).astype(np.float32)
            np.testing.assert_allclose(ReferenceEvaluator(m).run(None,{'input':x})[0],ReferenceEvaluator(n).run(None,{'input':x})[0],atol=1e-5,rtol=1e-5)

    def test_native_missing_bias_and_channel_blocks(self):
        from open_rknpu.native import compile_native_input
        m=onnx.load(ROOT/'native_input_blocks_suite/model023.onnx')
        for t in m.graph.initializer:
            if t.name==m.graph.node[0].input[2]:t.CopyFrom(nh.from_array(np.zeros(t.dims,np.float32),t.name))
        expected,_=compile_native_input(m)
        del m.graph.node[0].input[2]
        actual,_=compile_native_input(m)
        self.assertEqual(expected,actual)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.onnx';onnx.save(m,p)
            self.assertEqual(compile_sequence(p)[0],expected)
