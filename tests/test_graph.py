"""Format v5 named-tensor container and the two-head fan-out profile."""
from pathlib import Path
import os
import struct
import subprocess
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.model import checksum
from open_rknpu.sequence import (encode_sequence_v5,decode_sequence,tensor_native_bytes,
    LAYOUT_PACKED_U8,LAYOUT_NATIVE16,ROLE_INPUT,ROLE_OUTPUT,ROLE_INTERNAL)
from open_rknpu.scheduler import compile_sequence

ROOT=Path(__file__).resolve().parents[1]

def two_head_model(hidden=5,kernel_a=1,kernel_b=3,seed=7):
    rng=np.random.default_rng(seed)
    tensors=[
        nh.from_array(rng.uniform(-.7,.8,(hidden,3,1,1)).astype(np.float32),"w1"),
        nh.from_array(rng.uniform(-4,4,(hidden,)).astype(np.float32),"b1"),
        nh.from_array(rng.uniform(-.7,.8,(3,hidden,kernel_a,kernel_a)).astype(np.float32),"wa"),
        nh.from_array(rng.uniform(-4,4,(3,)).astype(np.float32),"ba"),
        nh.from_array(rng.uniform(-.7,.8,(3,hidden,kernel_b,kernel_b)).astype(np.float32),"wb"),
        nh.from_array(rng.uniform(-4,4,(3,)).astype(np.float32),"bb"),
    ]
    nodes=[
        h.make_node("Conv",["input","w1","b1"],["stem"],kernel_shape=[1,1]),
        h.make_node("Relu",["stem"],["relu1"]),
        h.make_node("Conv",["relu1","wa","ba"],["outputA"],kernel_shape=[kernel_a,kernel_a],pads=[kernel_a//2]*4),
        h.make_node("Conv",["relu1","wb","bb"],["outputB"],kernel_shape=[kernel_b,kernel_b],pads=[kernel_b//2]*4),
    ]
    graph=h.make_graph(nodes,"two_head",
        [h.make_tensor_value_info("input",1,[1,3,8,8])],
        [h.make_tensor_value_info("outputA",1,[1,3,8,8]),h.make_tensor_value_info("outputB",1,[1,3,8,8])],
        tensors)
    graph.value_info.append(h.make_tensor_value_info("relu1",1,[1,hidden,8,8]))
    model=h.make_model(graph,opset_imports=[h.make_opsetid("",13)]);model.ir_version=8
    return model

def write_model(directory,model,name="model.onnx"):
    path=Path(directory)/name
    onnx.save(model,path)
    return path

class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory()
        cls.binary=Path(cls.folder.name)/"inspect"
        subprocess.run([os.environ.get("CC","cc"),"-D_GNU_SOURCE","-std=c11","-Wall","-Wextra","-Werror",
                        str(ROOT/"runtime/open_rknpu.c"),str(ROOT/"runtime/main.c"),"-o",str(cls.binary)],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def check_both(self,data,valid):
        if valid:
            decode_sequence(data)
        else:
            with self.assertRaises(ValueError): decode_sequence(data)
        path=Path(self.folder.name)/"case.bin"
        path.write_bytes(data)
        result=subprocess.run([str(self.binary),"--inspect",str(path)],capture_output=True)
        self.assertEqual(result.returncode==0,valid,result.stderr.decode())

    def test_two_head_profile_is_v5_fanout(self):
        path=write_model(self.folder.name,two_head_model())
        data,meta=compile_sequence(path)
        info=decode_sequence(data)
        self.assertEqual(info["format_version"],5)
        self.assertEqual(info["task_count"],3)
        self.assertEqual(info["input_tensor_count"],1)
        self.assertEqual(info["output_tensor_count"],2)
        roles={t["name"]:t["role_name"] for t in info["tensors"]}
        self.assertEqual(roles,{"input0":"input","stem":"internal","output_a":"output","output_b":"output"})
        self.assertEqual([t["index"] for t in info["output_tensors"]],[0,1])
        self.assertEqual(meta["profile"],"two-head-fanout")
        stem=[t for t in info["tensors"] if t["name"]=="stem"][0]
        self.assertEqual((stem["layout_name"],stem["channels"]),( "native16",5))
        # Fan-out: one producer tensor, three tasks; the two heads read the same tensor.
        self.check_both(data,True)

    def test_two_head_head_kernels(self):
        for kernel_a,kernel_b in ((1,1),(1,3),(3,1),(3,3)):
            with self.subTest(kernel=kernel_a,kernel_b=kernel_b):
                path=write_model(self.folder.name,two_head_model(hidden=8,kernel_a=kernel_a,kernel_b=kernel_b,seed=11))
                data,_=compile_sequence(path)
                self.assertEqual(decode_sequence(data)["task_count"],3)
                self.check_both(data,True)

    def test_two_head_rejections(self):
        model=two_head_model()
        # Break the shared-stem contract: second head consumes the input directly.
        broken=onnx.load_from_string(model.SerializeToString())
        broken.graph.node[3].input[0]="input"
        path=write_model(self.folder.name,broken,"broken.onnx")
        with self.assertRaises(ValueError): compile_sequence(path)
        # Wrong hidden channel count.
        path=write_model(self.folder.name,two_head_model(hidden=2),"small.onnx")
        with self.assertRaises(ValueError): compile_sequence(path)

    def make_v5(self,outputs=2):
        payload=bytes(4096);tasks=[(0,126,29,768)]
        native=tensor_native_bytes(LAYOUT_NATIVE16,1,8,8,3)
        tensors=[
            dict(name="input0",role=ROLE_INPUT,layout=LAYOUT_PACKED_U8,index=0,shape=(1,8,8,3),
                 offset=8192,size=tensor_native_bytes(LAYOUT_PACKED_U8,1,8,8,3)),
            dict(name="hidden",role=ROLE_INTERNAL,layout=LAYOUT_NATIVE16,index=0,shape=(1,8,8,3),
                 offset=12288,size=native),
            dict(name="out0",role=ROLE_OUTPUT,layout=LAYOUT_NATIVE16,index=0,shape=(1,8,8,3),
                 offset=13312,size=native),
        ]
        if outputs==2:
            tensors.append(dict(name="out1",role=ROLE_OUTPUT,layout=LAYOUT_NATIVE16,index=1,
                shape=(1,4,4,3),offset=14336,size=tensor_native_bytes(LAYOUT_NATIVE16,1,4,4,3)))
        return encode_sequence_v5(payload,tensors=tensors,tasks=tasks,arena_bytes=16384,
            input_scale=1.0,input_zero_point=0,output_scale=0.5,output_zero_point=3,serial=True)

    def test_v5_roundtrip_and_rejections(self):
        data=self.make_v5()
        info=decode_sequence(data)
        self.assertEqual((info["format_version"],info["output_tensor_count"]),(5,2))
        self.assertEqual(info["output_tensors"][1]["shape_nhwc"] if "shape_nhwc" in info["output_tensors"][1] else
                         (info["output_tensors"][1]["batch"],info["output_tensors"][1]["height"],
                          info["output_tensors"][1]["width"],info["output_tensors"][1]["channels"]),(1,4,4,3))
        base=112+16
        cases=[("duplicate name",base+2*64,b"hidden\0\0"),("bad role",base+64+24,9),
               ("noncontiguous index",base+32,1),("bad layout",base+64+28,9),
               ("overlap payload",base+64+52,0)]
        for description,offset,value in cases:
            with self.subTest(description=description):
                bad=bytearray(data)
                if isinstance(value,bytes): bad[offset:offset+len(value)]=value
                else: struct.pack_into("<I",bad,offset,value)
                bad[80:84]=b"\0"*4;struct.pack_into("<I",bad,80,checksum(bad))
                self.check_both(bytes(bad),False)
        for truncated in (data[:95],data[:111],data[:-1],data+b"x"):
            with self.subTest(length=len(truncated)): self.check_both(truncated,False)

    def test_v5_single_output_roundtrip(self):
        self.check_both(self.make_v5(outputs=1),True)

if __name__=="__main__":
    unittest.main()
