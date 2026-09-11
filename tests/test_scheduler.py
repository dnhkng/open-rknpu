"""Graph and memory-layout boundaries for explicit sequence compilation."""
from pathlib import Path
import tempfile
import unittest
import onnx
from onnx import helper as h
from open_rknpu.scheduler import compile_sequence
from open_rknpu.model import decode

ROOT=Path(__file__).resolve().parents[1]


class SchedulerTests(unittest.TestCase):
    def test_trained_prefix_dimensions_and_tasks(self):
        data,_=compile_sequence(ROOT/"research/mnist_pool_suite/model000.onnx",.25,135)
        info=decode(data)
        self.assertEqual(info["shape_nhwc"],[1,28,28,1])
        self.assertEqual(info["output_shape_nhwc"],[1,14,14,8])
        self.assertEqual([t["enable"] for t in info["tasks"]],[29,96])
        self.assertTrue(info["serial"])
        self.assertLessEqual(info["payload_bytes"],info["input_offset"])
        self.assertLessEqual(info["output_offset"]+14*14*16,info["arena_bytes"])

    def test_reject_unsupported_attributes_and_connections(self):
        for variant in ("stride","padding","output","branch"):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                model=onnx.load(ROOT/"research/mnist_pool_suite/model000.onnx")
                pool=model.graph.node[-1]
                if variant in ("stride","padding"):
                    name="strides" if variant=="stride" else "pads"
                    kept=[a for a in pool.attribute if a.name!=name]
                    del pool.attribute[:];pool.attribute.extend(kept)
                    pool.attribute.append(h.make_attribute(name,[1,1] if variant=="stride" else [1]*4))
                elif variant=="output":
                    model.graph.output[0].type.tensor_type.shape.dim[2].dim_value=13
                else:
                    pool.input[0]=model.graph.node[0].output[0]
                path=Path(tmp)/"bad.onnx";onnx.save(model,path)
                with self.assertRaises((ValueError,onnx.shape_inference.InferenceError)):
                    compile_sequence(path)
