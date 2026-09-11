"""SPDX-License-Identifier: MIT
Train the tiny mel-CNN spoken-digit model and export the NPU graph.

Architecture (3x32x32 mel/delta input, 10 spoken-digit classes), chosen so that every
layer is an already board-verified open-rknpu primitive and the *whole* model runs on the
NPU (`open_rknpu.walk`, chains with pools):

    Conv(3->16,  3x3, pad1) -> BatchNorm -> Relu
    Conv(16->16, 3x3, pad1) -> BatchNorm -> Relu -> MaxPool(2x2/2)     # 16x16x16
    Conv(16->16, 3x3, pad1) -> BatchNorm -> Relu
    Conv(16->16, 3x3, pad1) -> BatchNorm -> Relu -> MaxPool(2x2/2)     # 8x8x16
    Conv(16->10, 1x1)                                                  # 8x8x10 logit map

BatchNorm is a *training* aid only: before export every BN is folded into the preceding
convolution (an exact affine map in eval mode), so the exported graph contains only
Conv/Relu/MaxPool. Classification averages the 8x8 logit map on the host (global average
pooling is linear, so it is exact; `ReduceMean` is not an open-rknpu primitive).

    PYTHONPATH=src python examples/mel-kws/train.py [--epochs 60]

Writes `build/features.npz`, `build/model.onnx` and `build/train_report.json`. PyTorch is a
training dependency only; the NPU path needs just NumPy and ONNX.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/mel-kws"))
import features as F  # noqa: E402

RECORDINGS = ROOT / "research/pretrained/fsdd/recordings"
OUT = Path(__file__).resolve().parent / "build"


def block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.BatchNorm2d(out_channels), nn.ReLU())


class MelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            block(3, 16), block(16, 16), nn.MaxPool2d(2),
            block(16, 16), block(16, 16), nn.MaxPool2d(2),
            nn.Conv2d(16, 10, 1),
        )

    def forward(self, x):
        return self.body(x)


def fold(model):
    """Return an equivalent Conv/Relu/MaxPool-only module with every BN folded."""
    model.eval()
    layers = []
    for layer in model.body:
        if isinstance(layer, nn.BatchNorm2d):
            conv = layers[-1]
            scale = layer.weight.detach() / torch.sqrt(layer.running_var.detach() + layer.eps)
            conv.weight.data = conv.weight.detach() * scale[:, None, None, None]
            conv.bias.data = ((conv.bias.detach() - layer.running_mean.detach()) * scale
                              + layer.bias.detach())
        else:
            layers.append(layer)
    return nn.Sequential(*layers)


def build_cache(force=False):
    """The UINT8 feature images for all 3,000 recordings, plus labels and the split."""
    path = OUT / "features.npz"
    if path.is_file() and not force:
        cached = np.load(path)
        return cached["x"], cached["y"], cached["test"]
    train, test = F.split(RECORDINGS)
    items = train + test
    filters = F.mel_filterbank()
    # Train on the exact bytes the runtime feeds the NPU (float32 in [0, 255]): the
    # input then has no quantization error of its own, and `to_uint8` is the single
    # source of truth for both training and the board input files.
    x = np.stack([F.to_uint8(F.features(item[0], filters)).astype(np.float32) for item in items])
    y = np.array([item[1] for item in items], dtype=np.int64)
    is_test = np.array([False] * len(train) + [True] * len(test))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, test=is_test)
    return x, y, is_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=110911)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    started = time.time()
    x, y, is_test = build_cache(args.force_cache)
    xt = torch.from_numpy(x[~is_test])
    yt = torch.from_numpy(y[~is_test])
    xv = torch.from_numpy(x[is_test])
    yv = torch.from_numpy(y[is_test])
    print(f"features {x.shape} train {len(xt)} test {len(xv)} ({time.time() - started:.1f}s)")
    np.save(OUT / "test_inputs.npy", x[is_test])
    np.save(OUT / "test_labels.npy", y[is_test])

    model = MelCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xt))
        total = 0.0
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[index]).mean(dim=(2, 3)), yt[index])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(index)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            accuracy = float((model(xv).mean(dim=(2, 3)).argmax(dim=1) == yv).float().mean())
        history.append(dict(epoch=epoch, loss=total / len(xt), test_accuracy=accuracy))
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  loss {total / len(xt):.4f}  test accuracy {accuracy * 100:.2f}%")

    # Fold BatchNorm, then check the folded model is numerically identical.
    model.eval()
    with torch.no_grad():
        before = model(xv[:16]).numpy()
    folded = fold(model).eval()
    with torch.no_grad():
        after = folded(xv[:16]).numpy()
    fold_delta = float(np.abs(before - after).max())
    if fold_delta > 1e-4:
        raise SystemExit(f"BatchNorm folding changed the logits by {fold_delta:.3e}")

    onnx_path = OUT / "model.onnx"
    torch.onnx.export(folded, torch.zeros(1, 3, 32, 32), str(onnx_path),
                      input_names=["input"], output_names=["output"], opset_version=13)

    from onnx.reference import ReferenceEvaluator
    import onnx
    session = ReferenceEvaluator(onnx.load(onnx_path))
    sample = xv[:16].numpy()
    exported = np.stack([session.run(None, {"input": row[None]})[0][0] for row in sample])
    onnx_delta = float(np.abs(exported - after).max())
    print(f"fold-vs-BN max abs delta {fold_delta:.3e}; ONNX-vs-folded {onnx_delta:.3e}")
    if onnx_delta > 1e-4:
        raise SystemExit("the exported ONNX does not match the trained model")

    (OUT / "train_report.json").write_text(json.dumps({
        "architecture": [str(layer) for layer in folded],
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "epochs": args.epochs, "batch": args.batch, "seed": args.seed, "lr": args.lr,
        "train_samples": int((~is_test).sum()), "test_samples": int(is_test.sum()),
        "float_test_accuracy": history[-1]["test_accuracy"],
        "batchnorm_fold_delta": fold_delta, "onnx_export_delta": onnx_delta,
        "history": history,
        "feature": {"window": F.WINDOW, "hop": F.HOP, "fft": F.FFT, "mels": F.MELS,
                    "samples": F.SAMPLES, "clip": F.CLIP},
    }, indent=2) + "\n")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes), "
          f"float test accuracy {history[-1]['test_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
