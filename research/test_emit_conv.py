"""Host checks for unsafe/out-of-profile graph acceptance and buffer construction."""
from pathlib import Path
import struct
import tempfile
import unittest
import onnx
from emit_conv import compile_model

ROOT=Path(__file__).resolve().parent

class CompilerTests(unittest.TestCase):
    def compile_modified(self, change):
        model=onnx.load(ROOT/"fixtures/identity/model.onnx")
        change(model)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model.onnx"
            onnx.save(model,path)
            return compile_model(path)

    def test_heldout_shape_and_weight_encoding(self):
        data,meta=compile_model(ROOT/"generated/heldout.onnx")
        self.assertEqual(meta["shape_nhwc"],[1,6,5,3])
        self.assertEqual(meta["input_channel_for_output"],[1,2,0])
        registers={}
        for i in range(meta["register_count"]):
            word=struct.unpack_from("<Q",data,8*i)[0]
            registers[word&65535]=(word>>16)&0xffffffff
        self.assertEqual(registers[0x1020],5<<16|6)
        self.assertEqual(registers[0x4024],480)
        self.assertEqual(registers[0x1110],0x440)
        self.assertEqual(data[0x440:0x44c],bytes([128,127,128,128,128,128,127,128,127,128,128,128]))
        self.assertEqual(len(data),8192)

    def test_reject_stride(self):
        def change(model):
            model.graph.node[0].attribute.append(onnx.helper.make_attribute("strides",[2,2]))
        with self.assertRaisesRegex(ValueError,"attribute"):
            self.compile_modified(change)

    def test_reject_dynamic_shape(self):
        def change(model):
            model.graph.input[0].type.tensor_type.shape.dim[2].dim_param="height"
        with self.assertRaisesRegex(ValueError,"static"):
            self.compile_modified(change)

    def test_reject_nonfinite_weights(self):
        def change(model):
            import numpy as np
            weights=onnx.numpy_helper.to_array(model.graph.initializer[0]).copy()
            weights[0,0,0,0]=np.float32(np.nan)
            model.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(weights,"weights"))
        with self.assertRaisesRegex(ValueError,"finite"):
            self.compile_modified(change)

if __name__=="__main__":
    unittest.main()
