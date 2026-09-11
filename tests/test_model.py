"""Model validation parity between the Python compiler and native C loader."""
from pathlib import Path
import os
import struct
import subprocess
import tempfile
import unittest
from open_rknpu.compiler import compile_model
from open_rknpu.model import encode,decode,checksum

ROOT=Path(__file__).resolve().parents[1]

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload,metadata=compile_model(ROOT/"research/generated/k3relu_heldout.onnx")
        cls.good=encode(payload,metadata)
        cls.folder=tempfile.TemporaryDirectory()
        cls.binary=Path(cls.folder.name)/"inspect"
        subprocess.run([os.environ.get("CC","cc"),"-D_GNU_SOURCE","-std=c11","-Wall","-Wextra","-Werror",
                        str(ROOT/"runtime/open_rknpu.c"),str(ROOT/"runtime/main.c"),"-o",str(cls.binary)],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def check_both(self,data,valid,shape=(1,7,6,3)):
        if valid:
            info=decode(data)
            self.assertEqual(info["shape_nhwc"],list(shape))
        else:
            with self.assertRaises(ValueError): decode(data)
        path=Path(self.folder.name)/"case.bin"
        path.write_bytes(data)
        result=subprocess.run([str(self.binary),"--inspect",str(path)],capture_output=True)
        self.assertEqual(result.returncode==0,valid,result.stderr.decode())

    def test_valid_container(self):
        self.check_both(self.good,True)

    def test_wide_output_container(self):
        payload,meta=compile_model(ROOT/"research/wide_suite/model028.onnx")
        data=encode(payload,meta)
        self.check_both(data,True,meta["shape_nhwc"])
        info=decode(data)
        self.assertEqual(info["output_shape_nhwc"],[1,5,6,16])
        self.assertEqual(info["output_bytes"],480)

    def test_wide_spatial_convolution(self):
        for kernel in (3,5):
            payload,meta=compile_model(ROOT/f"research/wide_k{kernel}_suite/model028.onnx")
            data=encode(payload,meta)
            self.check_both(data,True,meta["shape_nhwc"])
            self.assertEqual((decode(data)["kernel_size"],decode(data)["output_bytes"]),(kernel,480))

    def test_grayscale_container(self):
        for kernel in (1,3,5):
            payload,meta=compile_model(ROOT/f"research/gray_k{kernel}_suite/model028.onnx")
            data=encode(payload,meta)
            self.check_both(data,True,(1,5,6,1))
            self.assertEqual(decode(data)["input_bytes"],30)

    def test_large_grayscale_layout(self):
        payload,meta=compile_model(ROOT/"research/gray_large_suite/model020.onnx")
        good=encode(payload,meta)
        self.check_both(good,True,(1,28,28,1))
        self.assertEqual(decode(good)["arena_bytes"],28672)
        self.assertEqual(decode(good)["input_bytes"],784)
        for offset,value in ((24,33),(28,33),(32,3),(40,16),(64,16384)):
            data=bytearray(good);struct.pack_into("<I",data,offset,value)
            data[84:88]=b"\0"*4;struct.pack_into("<I",data,84,checksum(data))
            self.check_both(data,False)

    def test_affine_input_container(self):
        path=ROOT/"research/mnist_first_suite/model000.onnx"
        payload,meta=compile_model(path,input_scale=.25,input_zero_point=135)
        good=encode(payload,meta)
        self.check_both(good,True,(1,28,28,1))
        info=decode(good)
        self.assertEqual((info["format_version"],info["input_scale"],info["input_zero_point"]),(2,.25,135))
        for offset,value in ((8,1),(88,0),(88,0x7fc00000),(88,0xbf800000),(92,256)):
            data=bytearray(good);struct.pack_into("<I",data,offset,value)
            data[84:88]=b"\0"*4;struct.pack_into("<I",data,84,checksum(data))
            self.check_both(data,False)
        for scale,zp in ((0,0),(-1,0),(float("nan"),0),(1,-1),(1,256),(1,.5)):
            with self.assertRaises(ValueError):
                compile_model(path,input_scale=scale,input_zero_point=zp)

    def test_sequence_container_and_bounds(self):
        good=(ROOT/"research/sequence_suite/model000.bin").read_bytes()
        self.check_both(good,True,(1,14,14,1))
        info=decode(good)
        self.assertEqual((info["format_version"],info["task_count"],info["output_bytes"]),(3,2,3136))
        # Recompute checksum so every rejection exercises semantic validation.
        for offset,value in ((8,4),(16,0),(24,16),(40,15),(44,0),(48,4096),
                             (52,0),(56,16384),(60,65),(64,0x7fc00000),(68,256),
                             (80,0),(84,4),(96,16384),(100,257),(104,0),(108,0)):
            data=bytearray(good);struct.pack_into("<I",data,offset,value)
            data[80:84]=b"\0"*4;struct.pack_into("<I",data,80,checksum(data))
            if offset==80:
                data[-1]^=1
            self.check_both(data,False)
        two=decode((ROOT/'research/standalone_mul_suite/model000.bin').read_bytes())
        self.assertEqual(two['input_tensor_count'],2)
        for data in (good[:95],good[:-1],good+b"x"):
            self.check_both(data,False)

    def test_v4_mutable_constant_descriptor(self):
        from open_rknpu.scheduler import compile_sequence
        from open_rknpu.sequence import decode_sequence
        data,meta=compile_sequence(ROOT/'research/native_input_suite/model000.onnx',mutable_weights=True)
        info=decode_sequence(data)
        self.assertEqual((info['format_version'],info['constant_count'],meta['mutable_constants']),(4,1,['conv.parameters']))
        self.assertEqual(info['constants'][0]['kind'],1)
        self.check_both(data,True,info['shape_nhwc'])
        # Descriptor offset starts after the single task table.
        bad=bytearray(data);struct.pack_into('<I',bad,96+16+28,0)
        bad[80:84]=b'\0'*4;struct.pack_into('<I',bad,80,checksum(bad))
        self.check_both(bad,False)
        mul,_=compile_sequence(ROOT/'research/mul_broadcast_suite/model001.onnx',mutable_constants=True)
        minfo=decode_sequence(mul)
        self.assertEqual((minfo['format_version'],minfo['constants'][0]['name'],minfo['constants'][0]['kind']),(4,'mul.factor',3))

    def test_native_input_layout_and_bounds(self):
        good=(ROOT/"research/native_combined_suite/model013.bin").read_bytes()
        shape=decode(good)["shape_nhwc"]
        self.check_both(good,True,shape)
        self.assertEqual(decode(good)["input_layout"],"native16")
        for offset,value in ((88,0),(88,2),(92,1),(24,0),(24,33),(40,16),(56,4096),(48,4096)):
            with self.subTest(offset=offset,value=value):
                data=bytearray(good);struct.pack_into("<I",data,offset,value)
                data[80:84]=b"\0"*4;struct.pack_into("<I",data,80,checksum(data))
                self.check_both(data,False)

    def test_native_multiple_channel_blocks(self):
        good=(ROOT/"research/native_input_blocks_suite/model023.bin").read_bytes()
        self.check_both(good,True,(1, 8, 6, 32))
        info=decode(good)
        self.assertEqual(info["output_shape_nhwc"], [1, 8, 6, 64])
        for offset,value in ((24,33),(36,129),(56,info["input_offset"]+64),(48,4096)):
            data=bytearray(good);struct.pack_into("<I",data,offset,value)
            data[80:84]=b"\0"*4;struct.pack_into("<I",data,80,checksum(data))
            self.check_both(data,False)

    def test_two_task_container_and_rejections(self):
        payload,meta=compile_model(ROOT/"research/generated/chain_heldout4.onnx")
        good=encode(payload,meta)
        self.check_both(good,True,(1,8,8,3))
        self.assertEqual(decode(good)["task_count"],2)
        for offset,value in ((20,99),(24,7),(28,7),(32,1),(36,4),(68,5),(68,7),(72,0),(88,2)):
            with self.subTest(offset=offset):
                data=bytearray(good)
                struct.pack_into("<I",data,offset,value)
                data[84:88]=b"\0"*4
                struct.pack_into("<I",data,84,checksum(data))
                self.check_both(data,False)

    def test_pool_container_dimensions_and_invalid_shapes(self):
        for kind,profile in (("max",3),("average",4)):
            payload,meta=compile_model(ROOT/f"research/generated/pool_{kind}_r1.onnx")
            good=encode(payload,meta)
            self.check_both(good,True,(1,8,8,3))
            info=decode(good)
            self.assertEqual((info["profile"],info["output_bytes"]),(profile,48))
            self.assertEqual(info["output_shape_nhwc"],[1,4,4,3])
            for offset,value in ((24,7),(28,7),(36,4),(68,3)):
                data=bytearray(good);struct.pack_into("<I",data,offset,value)
                data[84:88]=b"\0"*4;struct.pack_into("<I",data,84,checksum(data))
                self.check_both(data,False)

    def test_staged_pool_container(self):
        for kind,profile in (("max",5),("average",6)):
            payload,meta=compile_model(ROOT/f"research/generated/reduce_{kind}_r1.onnx")
            data=encode(payload,meta)
            self.check_both(data,True,(1,8,8,3))
            info=decode(data)
            self.assertEqual((info["profile"],info["task_count"],info["output_bytes"]),(profile,4,3))
            self.assertEqual(info["output_shape_nhwc"],[1,1,1,3])

    def test_spatial_network_container(self):
        for index,profile in ((26,7),(27,8)):
            payload,meta=compile_model(ROOT/f"research/network_suite/model{index:03}.onnx")
            data=encode(payload,meta);self.check_both(data,True,(1,8,8,3))
            info=decode(data)
            self.assertEqual((info["profile"],info["task_count"],info["output_bytes"]),(profile,5,3))

    def test_truncated_and_trailing(self):
        for data in (self.good[:12],self.good[:-1],self.good+b"extra"):
            with self.subTest(length=len(data)): self.check_both(data,False)

    def test_corrupted_payload(self):
        data=bytearray(self.good)
        data[200]^=1
        self.check_both(data,False)

    def test_invalid_fields_even_with_correct_checksum(self):
        for offset,value in ((8,99),(16,3588),(24,65535),(32,16),(36,0),(36,17),(48,0xffffffff),(56,0xffffffff),(68,7),(72,2),(76,128),(80,0x7fc00000),(88,1)):
            with self.subTest(offset=offset,value=value):
                data=bytearray(self.good)
                struct.pack_into("<I",data,offset,value)
                data[84:88]=b"\0"*4
                struct.pack_into("<I",data,84,checksum(data))
                self.check_both(data,False)
