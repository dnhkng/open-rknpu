"""SPDX-License-Identifier: MIT"""
import argparse
import json
from pathlib import Path
from . import __version__
from .compiler import compile_model
from .model import encode,decode

def main():
    parser=argparse.ArgumentParser(prog="open-rknpu")
    parser.add_argument("--version",action="version",version=__version__)
    sub=parser.add_subparsers(dest="command",required=True)
    compile_parser=sub.add_parser("compile",help="compile a supported static ONNX convolution graph")
    compile_parser.add_argument("model",type=Path)
    compile_parser.add_argument("--target",choices=["rv1103"],default="rv1103")
    compile_parser.add_argument("--quantize",choices=["int8"],default="int8")
    compile_parser.add_argument("--output-scale",type=float)
    compile_parser.add_argument("--output-zero-point",type=int)
    compile_parser.add_argument("--input-scale",type=float,default=1.0)
    compile_parser.add_argument("--input-zero-point",type=int,default=0)
    compile_parser.add_argument("--mul-a-zero-point",type=int,default=0)
    compile_parser.add_argument("--mul-b-zero-point",type=int,default=0)
    compile_parser.add_argument("--sequence",action="store_true",help="use expanded Conv, elementwise, depthwise and activation task-table profiles")
    compile_parser.add_argument("--mutable-weights",action="store_true",help="emit a v4 packed Conv parameter descriptor")
    compile_parser.add_argument("--mutable-constants",action="store_true",help="emit a v4 packed Mul factor descriptor")
    compile_parser.add_argument("--expose-intermediates",action="store_true",help="emit v5 chain intermediates as external outputs")
    compile_parser.add_argument("--reuse-intermediates",action="store_true",help="reuse chain intermediate arena buffers by lifetime")
    compile_parser.add_argument("--asymmetric-depthwise",action="store_true",help="use per-channel asymmetric depthwise weight zero points")
    compile_parser.add_argument("--per-channel-mul",action="store_true",help="quantize an immutable Mul constant per channel via a 1x1 depthwise Conv")
    compile_parser.add_argument("--tiles",type=int,help="split an 8x8 1x1 Conv chain into N height strips")
    compile_parser.add_argument("--submission",choices=["serial","batched"],default="serial",
        help="'batched' links a single-engine task list into one job (deep chains only); serial is the default")
    compile_parser.add_argument("--calibration",type=Path,help="directory of uint8/float32 NCHW .npy calibration batches")
    compile_parser.add_argument("-o","--output",type=Path,required=True)
    inspect_parser=sub.add_parser("inspect",help="validate and inspect an executable")
    inspect_parser.add_argument("model",type=Path)
    normalize_parser=sub.add_parser("normalize",help="fold constant reshapes and convolution bias/padding in ONNX")
    normalize_parser.add_argument("model",type=Path)
    normalize_parser.add_argument("-o","--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=="inspect":
            print(json.dumps(decode(args.model.read_bytes()),indent=2))
        elif args.command=="normalize":
            import onnx
            from .normalize import normalize_model
            model=normalize_model(onnx.load(args.model))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            onnx.save(model,args.output)
            print("Normalized %s -> %s (%d nodes)"%(args.model,args.output,len(model.graph.node)))
        else:
            if (args.output_scale is None)!=(args.output_zero_point is None):
                raise ValueError("specify output scale and zero point together")
            report=None
            if args.calibration:
                if (args.input_scale,args.input_zero_point)!=(1.0,0):
                    raise ValueError("calibration with input quantization overrides is not supported yet")
                if args.output_scale is not None or args.output_zero_point is not None:
                    raise ValueError("calibration and output quantization overrides cannot be combined")
                from .calibration import measure
                report=measure(args.model,args.calibration)
            if args.sequence:
                from .scheduler import compile_sequence
                output_range=None if args.output_scale is None else dict(scale=args.output_scale,zero_point=args.output_zero_point)
                data,metadata=compile_sequence(args.model,args.input_scale,args.input_zero_point,output_range,
                                               (args.mul_a_zero_point,args.mul_b_zero_point),args.mutable_weights,args.mutable_constants,
                                               None if report is None else report["ranges"],args.expose_intermediates,args.reuse_intermediates,args.asymmetric_depthwise,
                                               args.per_channel_mul,args.submission,args.tiles)
            else:
                if args.mutable_weights or args.mutable_constants:raise ValueError('mutable parameters require --sequence')
                if (args.mul_a_zero_point,args.mul_b_zero_point)!=(0,0):
                    raise ValueError("Mul operand zero points require --sequence")
                payload,metadata=compile_model(args.model,args.output_scale,args.output_zero_point,
                                               None if report is None else report["ranges"],
                                               args.input_scale,args.input_zero_point)
                data=encode(payload,metadata)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes(data)
            if report is not None:
                args.output.with_suffix(".calibration.json").write_text(json.dumps(report,indent=2)+"\n")
            mode=metadata.get("submission","serial")
            runs=metadata.get("engine_runs")
            detail="" if not runs or len(runs)<2 else " in %d engine runs"%len(runs)
            print("Compiled %s -> %s (%d bytes, %s submission%s)"%(args.model,args.output,len(data),mode,detail))
    except (ValueError,KeyError,OSError) as exc:
        parser.exit(1,"open-rknpu: %s\n"%exc)
