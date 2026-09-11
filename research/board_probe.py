"""Small ctypes vendor-reference probe; run on ARM32 board, no numpy needed."""
import ctypes as C
import json
import os
import sys

U = C.c_uint32
class Attr(C.Structure):
    _fields_ = [("index", U), ("n_dims", U), ("dims", U * 16),
                ("name", C.c_char * 256), ("n_elems", U), ("size", U),
                ("fmt", U), ("type", U), ("qnt_type", U), ("fl", C.c_int8),
                ("zp", C.c_int32), ("scale", C.c_float), ("w_stride", U),
                ("size_with_stride", U), ("pass_through", C.c_uint8), ("h_stride", U)]
class Mem(C.Structure):
    _fields_ = [("virt_addr", C.c_void_p), ("phys_addr", C.c_uint64),
                ("fd", C.c_int32), ("offset", C.c_int32), ("size", U),
                ("flags", U), ("priv_data", C.c_void_p)]

def check(rc, operation):
    if rc != 0:
        raise RuntimeError("%s returned %d" % (operation, rc))

def describe(a):
    return {"name": a.name.decode(), "dims": list(a.dims)[:a.n_dims],
            **{k: getattr(a, k) for k in ("size", "size_with_stride", "fmt", "type", "zp", "scale", "w_stride")}}

def main():
    assert C.sizeof(C.c_void_p) == 4, "ARM32 board only"
    assert C.sizeof(Attr) == 376
    lib = C.CDLL(sys.argv[1])
    ctx = U()
    specs = {
        "rknn_init": [C.POINTER(U), C.c_void_p, U, U, C.c_void_p],
        "rknn_query": [U, U, C.c_void_p, U],
        "rknn_create_mem": [U, U],
        "rknn_set_io_mem": [U, C.POINTER(Mem), C.POINTER(Attr)],
        "rknn_mem_sync": [U, C.POINTER(Mem), U],
        "rknn_run": [U, C.c_void_p],
        "rknn_destroy_mem": [U, C.POINTER(Mem)],
        "rknn_destroy": [U],
    }
    for name, args in specs.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = C.POINTER(Mem) if name == "rknn_create_mem" else C.c_int
    model = C.create_string_buffer(open(sys.argv[2], "rb").read())
    check(lib.rknn_init(C.byref(ctx), model, len(model)-1, 0, None), "init")
    mems = []
    try:
        attrs = {}
        for command in (1, 2, 8, 9):
            attr = Attr()
            check(lib.rknn_query(ctx, command, C.byref(attr), C.sizeof(attr)), "query %d" % command)
            attrs[command] = attr
            print("ATTR", command, json.dumps(describe(attr)), flush=True)
        a, b = attrs[1], attrs[9]
        a.type, a.fmt, a.pass_through = 3, 1, 0  # documented UINT8 NHWC path
        for attr in (a, b):
            mem = lib.rknn_create_mem(ctx, attr.size_with_stride)
            if not mem:
                raise RuntimeError("create_mem failed")
            mems.append(mem)
            print("MEM", mem.contents.size, hex(mem.contents.phys_addr), flush=True)
            check(lib.rknn_set_io_mem(ctx, mem, C.byref(attr)), "set_io_mem")
        stride = a.w_stride or 8
        assert stride * 8 * 3 <= mems[0].contents.size
        results = []
        for case in ("zero", "constant", "ramp", "impulse"):
            C.memset(mems[0].contents.virt_addr, 0, mems[0].contents.size)
            data = (C.c_uint8 * mems[0].contents.size).from_address(mems[0].contents.virt_addr)
            for h in range(8):
                for w in range(8):
                    for c in range(3):
                        value = 0
                        if case == "constant": value = 128
                        if case == "ramp": value = (h * 24 + w * 3 + c)
                        if case == "impulse" and (h, w, c) == (3, 4, 1): value = 255
                        data[(h * stride + w) * 3 + c] = value
            check(lib.rknn_mem_sync(ctx, mems[0], 1), "input sync")
            check(lib.rknn_run(ctx, None), "run")
            check(lib.rknn_mem_sync(ctx, mems[1], 2), "output sync")
            result = C.string_at(mems[1].contents.virt_addr, b.size_with_stride)
            results.append({"case": case, "output_hex": result.hex()})
        print("RESULT", json.dumps({"output_attr": describe(b), "cases": results}), flush=True)
    finally:
        for mem in reversed(mems):
            check(lib.rknn_destroy_mem(ctx, mem), "destroy_mem")
        check(lib.rknn_destroy(ctx), "destroy")

if __name__ == "__main__":
    main()
