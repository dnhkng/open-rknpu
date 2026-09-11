"""Calibration data contracts, sequence calibration and accuracy reporting."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
from open_rknpu.calibration import _histogram, kl_range, measure, percentile_range
from open_rknpu.compiler import compile_model
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.accuracy import error_metrics,classification_accuracy,layerwise_quantization_error

MODEL=Path(__file__).resolve().parents[1]/"research/fixtures/identity/model.onnx"

def sequence_model(hidden=5,seed=3):
    rng=np.random.default_rng(seed)
    w1=rng.uniform(-.7,.8,(hidden,3,1,1)).astype(np.float32);b1=rng.uniform(-2,2,(hidden,)).astype(np.float32)
    w2=rng.uniform(-.7,.8,(3,hidden,1,1)).astype(np.float32);b2=rng.uniform(-2,2,(3,)).astype(np.float32)
    nodes=[h.make_node("Conv",["input","w1","b1"],["c1"],kernel_shape=[1,1]),
           h.make_node("Relu",["c1"],["r1"]),
           h.make_node("Conv",["r1","w2","b2"],["output"],kernel_shape=[1,1])]
    graph=h.make_graph(nodes,"seq",[h.make_tensor_value_info("input",1,[1,3,8,8])],
        [h.make_tensor_value_info("output",1,[1,3,8,8])],
        [nh.from_array(w1,"w1"),nh.from_array(b1,"b1"),nh.from_array(w2,"w2"),nh.from_array(b2,"b2")])
    graph.value_info.append(h.make_tensor_value_info("r1",1,[1,hidden,8,8]))
    model=h.make_model(graph,opset_imports=[h.make_opsetid("",13)]);model.ir_version=8
    return model

class CalibrationTests(unittest.TestCase):
    def test_known_activation_range_and_batch_count(self):
        with tempfile.TemporaryDirectory() as folder:
            samples=np.zeros((2,3,8,8),np.uint8);samples[1]=128
            np.save(Path(folder)/"batch.npy",samples)
            report=measure(MODEL,folder)
            self.assertEqual(report["samples"],2)
            entry=report["ranges"]["output"]
            self.assertEqual((entry["min"],entry["max"]),(0,96))
            self.assertEqual(entry["zero_point"],-128)
            _,meta=compile_model(MODEL,calibration_ranges=report["ranges"])
            self.assertAlmostEqual(meta["output_scale"],96/255,places=7)
            with self.assertRaisesRegex(ValueError,"cannot be combined"):
                compile_model(MODEL,1,-128,report["ranges"])

    def test_reject_empty_bad_shape_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,".npy"):
                measure(MODEL,folder)
            for data in (np.zeros((8,8,3),np.uint8),np.full((1,3,8,8),np.nan,np.float32),
                         np.full((1,3,8,8),256,np.float32)):
                np.save(Path(folder)/"batch.npy",data)
                with self.assertRaises(ValueError):measure(MODEL,folder)

    def test_sequence_calibration_uses_measured_ranges(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"seq.onnx";onnx.save(sequence_model(),path)
            rng=np.random.default_rng(9)
            np.save(Path(folder)/"cal.npy",rng.integers(0,256,(16,3,8,8),dtype=np.uint8))
            report=measure(path,folder)
            self.assertEqual(sorted(report["ranges"]),["output","r1"])
            analytic,_=compile_sequence(path)
            calibrated,meta=compile_sequence(path,calibration_ranges=report["ranges"])
            self.assertNotEqual(decode_sequence(analytic)["output_scale"],decode_sequence(calibrated)["output_scale"])
            self.assertEqual(decode_sequence(calibrated)["output_scale"],meta["output_scale"])
            with self.assertRaisesRegex(ValueError,"cannot be combined"):
                compile_sequence(path,output_range={"scale":1.0,"zero_point":0},calibration_ranges=report["ranges"])

    def test_calibration_requires_a_measured_tensor(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"seq.onnx";onnx.save(sequence_model(),path)
            with self.assertRaisesRegex(ValueError,"lack tensor"):
                compile_sequence(path,calibration_ranges={"something":{"scale":1.0,"zero_point":0}})

    def test_accuracy_metrics_and_classification(self):
        integer=np.array([[[[0,1],[2,3]]]],np.int8)
        reference=np.array([[[[0.0,1.0],[2.0,3.0]]]],np.float64)
        metrics=error_metrics(integer,reference,scale=1.0,zero_point=0)
        self.assertEqual((metrics["values"],metrics["mae"],metrics["max_error"]),(4,0.0,0.0))
        shifted=error_metrics(integer,reference,scale=2.0,zero_point=0)
        self.assertAlmostEqual(shifted["mae"],1.5)
        result=classification_accuracy(np.array([[0.1,0.9],[0.8,0.2],[0.3,0.7]]),np.array([1,0,1]))
        self.assertEqual((result["correct"],result["accuracy"]),(3,1.0))

    def test_layerwise_quantization_error(self):
        model=sequence_model()
        ranges={"r1":{"scale":0.01,"zero_point":0},"output":{"scale":0.02,"zero_point":0}}
        inputs=np.zeros((2,8,8,3),np.uint8)
        report=layerwise_quantization_error(model,ranges,inputs)
        self.assertEqual(report["samples"],2)
        self.assertEqual(sorted(report["layers"]),["output","r1"])
        self.assertTrue(all(entry["mae"]>=0 for entry in report["layers"].values()))

    def test_native_output_range_override_changes_scale(self):
        from open_rknpu.chain import native_quantize
        rng=np.random.default_rng(21)
        weights=rng.uniform(-.7,.8,(16,8,5,5)).astype(np.float32)
        bias=rng.uniform(-2,2,(16,)).astype(np.float32)
        analytic=native_quantize(weights,bias,1.0,0)
        calibrated=native_quantize(weights,bias,1.0,0,{"scale":0.13411250710487366,"zero_point":35})
        self.assertNotEqual(analytic.output_scale,calibrated.output_scale)
        self.assertEqual(calibrated.output_scale,0.13411250710487366)
        self.assertEqual(calibrated.output_zero_point,35)
        with self.assertRaises(ValueError):
            native_quantize(weights,bias,1.0,0,{"scale":0.0,"zero_point":0})

    def test_percentile_and_kl_range_selection(self):
        # A bulk in [0,10] with a 0.05% spike at 100: min/max spans all of it, while the
        # percentile tail cut and the KL search keep the resolution on the bulk.
        rng=np.random.default_rng(4)
        values=np.concatenate([rng.uniform(0,10,100000),np.full(50,100.0)])
        hist,edges=_histogram(values,2048,0,100)
        self.assertEqual(percentile_range(hist,edges,100.0),(0.0,100.0))
        low,high=percentile_range(hist,edges,99.9)
        self.assertEqual(low,0.0)
        self.assertLess(high,11.0)
        kl_low,kl_high=kl_range(hist,edges)
        self.assertEqual(kl_low,0.0)
        self.assertLess(kl_high,11.0)
        # The criterion is a proper divergence: never negative, and a threshold inside
        # the bulk is worse than one at the bulk boundary.
        def divergence(i):
            counts=hist.astype(np.float64);n=len(counts);levels=128
            reference=counts[:i].copy();reference[i-1]+=counts[i:].sum();reference/=reference.sum()
            index=np.arange(i);cuts=(index*levels//i).astype(int)
            widths=np.bincount(cuts,minlength=levels)
            mass=np.bincount(cuts,weights=counts[:i],minlength=levels);mass[-1]+=counts[i:].sum()
            full=np.zeros(n);full[:i]=mass[cuts]/np.maximum(widths[cuts],1);full/=full.sum()
            probability=counts/counts.sum();mask=probability>0
            return float(np.sum(probability[mask]*np.log(probability[mask]/np.maximum(full[mask],1e-12))))
        self.assertGreaterEqual(divergence(128),0.0)
        self.assertLess(divergence(205),divergence(128))
        # A uniform distribution has no tail to clip.
        uniform,uniform_edges=_histogram(rng.uniform(0,1,50000),2048,0,1)
        self.assertAlmostEqual(kl_range(uniform,uniform_edges)[1],1.0,places=3)

    def test_measure_methods_change_the_range_and_reject_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"seq.onnx";onnx.save(sequence_model(),path)
            rng=np.random.default_rng(11)
            cases=rng.integers(0,24,(24,3,8,8),dtype=np.uint8)
            cases[rng.random(cases.shape)<0.002]=255
            np.save(Path(folder)/"cal.npy",cases)
            reports={method:measure(path,folder,method=method)
                     for method in ("minmax","percentile","kl")}
            self.assertTrue(all(report["samples"]==24 for report in reports.values()))
            self.assertGreaterEqual(reports["minmax"]["ranges"]["output"]["max"],
                                    reports["percentile"]["ranges"]["output"]["max"])
            self.assertEqual(reports["percentile"]["percentile"],99.99)
            self.assertEqual(reports["kl"]["bins"],2048)
            for report in reports.values():
                binary,meta=compile_sequence(path,calibration_ranges=report["ranges"])
                self.assertEqual(decode_sequence(binary)["output_scale"],meta["output_scale"])
            self.assertNotEqual(reports["minmax"]["ranges"]["output"]["scale"],
                                reports["kl"]["ranges"]["output"]["scale"])
            with self.assertRaisesRegex(ValueError,"method must be one of"):
                measure(path,folder,method="entropy")

    def test_percentile_calibration_evidence_is_retained(self):
        suite=Path(__file__).resolve().parents[1]/"research"/"percentile_calibration_suite"
        manifest=json.loads((suite/"manifest.json").read_text())
        self.assertEqual([entry["method"] for entry in manifest],
                         ["analytic","minmax","percentile","kl"])
        by_method={entry["method"]:entry for entry in manifest}
        # Measured ranges beat the conservative analytic one on the bulk, and the
        # tail-clipping methods trade extreme accuracy for it (the documented trade-off).
        self.assertLess(by_method["minmax"]["bulk_mae"],by_method["analytic"]["bulk_mae"])
        self.assertLess(by_method["percentile"]["bulk_mae"],by_method["minmax"]["bulk_mae"])
        self.assertLess(by_method["kl"]["bulk_mae"],by_method["percentile"]["bulk_mae"])
        self.assertLess(by_method["analytic"]["extreme_max_error"],
                        by_method["kl"]["extreme_max_error"])
        results=json.loads((suite/"board_results_0.json").read_text())
        self.assertEqual(len(results),4)
        for record in results:
            with self.subTest(model=record["model"]):
                self.assertTrue(record["passed"])
                self.assertEqual(record["inferences"],19)
        self.assertTrue((suite/"board_summary.txt").is_file())


if __name__=="__main__":
    unittest.main()
