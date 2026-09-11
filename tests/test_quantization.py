"""Fixed-point and compiler boundaries backed by recorded hardware experiments."""
from pathlib import Path
import json
import unittest
import numpy as np
from open_rknpu.quantization import Quantization,reference,quantize

ROOT=Path(__file__).resolve().parents[1]/"research"

class QuantizationTests(unittest.TestCase):
    def test_signed_oracle_parameters(self):
        weights=np.array([[-.25,.5,.75],[.5,-.25,0],[0,.125,-.25]],np.float32)
        q=quantize(weights,[0,0,0],1.39068627,-83)
        np.testing.assert_array_equal(q.weights,[[-128,63,127],[127,-128,-43],[42,127,-128]])
        np.testing.assert_array_equal(q.biases,[32512,10880,-10880])
        self.assertEqual((q.multiplier,q.shift),(23655,23))

    def test_rounding_order_against_hardware(self):
        metadata=json.loads((ROOT/"generated/signed.json").read_text())
        params=metadata["quantization"]
        for key in ("weights","weight_zero_points","weight_scales","biases","channel_multipliers"):
            params[key]=np.array(params[key])
        q=Quantization(**params)
        # Exact random0 input produced by the C harness's specified LCG.
        inputs=np.zeros((8,8,3),dtype=np.uint8)
        state=(1103+4*0x9e3779b9)&0xffffffff
        for i in range(inputs.size):
            state=(1664525*state+1013904223)&0xffffffff
            inputs.flat[i]=state>>24
        line=next(x for x in (ROOT/"generated_signed.log").read_text().splitlines() if x.startswith("OUTPUT random0 "))
        observed=np.frombuffer(bytes.fromhex(line.split()[2]),dtype=np.int8).reshape(8,8,16)[:,:,:3]
        np.testing.assert_array_equal(reference(inputs,q),observed)
        self.assertGreater(np.count_nonzero(reference(inputs,q,"combined")!=observed),0)

    def test_nonfinite_and_overflow(self):
        with self.assertRaisesRegex(ValueError,"finite"):
            quantize(np.full((3,3),np.nan),[0,0,0])
        with self.assertRaisesRegex(ValueError,"overflow"):
            quantize(np.eye(3,dtype=np.float32),[1e8,0,0])

    def test_channel_halfway_rounds_to_even_on_hardware(self):
        inputs=np.arange(192,dtype=np.uint8).reshape(8,8,3)
        for sign,name in ((1,"ties_positive"),(-1,"ties_negative")):
            with self.subTest(name=name):
                q=quantize((np.diag([.25,.5,1])*sign).astype(np.float32),[0,0,0])
                line=next(x for x in (ROOT/f"generated_{name}.log").read_text().splitlines()
                          if x.startswith("OUTPUT ramp "))
                observed=np.frombuffer(bytes.fromhex(line.split()[2]),dtype=np.int8).reshape(8,8,16)[:,:,:3]
                np.testing.assert_array_equal(reference(inputs,q),observed)
                acc=np.einsum("hwc,oc->hwo",inputs.astype(np.int64)-128,
                              q.weights-q.weight_zero_points[:,None])+q.biases
                old_scaled=(acc*q.channel_multipliers+8192)>>14
                old=np.clip(((old_scaled*q.multiplier+(1<<(q.shift-1)))>>q.shift)
                            +q.output_zero_point,-128,127)
                self.assertGreater(np.count_nonzero(old!=observed),0)
