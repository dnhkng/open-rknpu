"""SPDX-License-Identifier: MIT
Fetch the MNIST and Fashion-MNIST IDX files used by the classifier examples.

The datasets are **not** redistributed with this repository. This script downloads the four
official gzip files into `research/pretrained/<dataset>/{data,test-data}/` and verifies their
sha256; the hashes below are the ones the recorded board/accuracy runs used.

    python examples/fetch_idx.py --dataset mnist
    python examples/fetch_idx.py --dataset fashion
    python examples/fetch_idx.py --dataset all --force     # re-download everything

MNIST: http://yann.lecun.com/exdb/mnist/ (mirrored on the S3 bucket below).
Fashion-MNIST: Zalando Research, MIT licensed.
"""
from pathlib import Path
import argparse
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
SPLIT = {FILES[0]: "data", FILES[1]: "data", FILES[2]: "test-data", FILES[3]: "test-data"}

DATASETS = {
    "mnist": {
        "base": "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "directory": "research/pretrained/mnist",
        "sha256": {
            "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
            "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
            "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
            "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        },
    },
    "fashion": {
        "base": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "directory": "research/pretrained/fashion-mnist",
        "sha256": {
            "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
            "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
            "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
            "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
        },
    },
}


def fetch(name, force=False):
    spec = DATASETS[name]
    folder = ROOT / spec["directory"]
    for filename in FILES:
        destination = folder / SPLIT[filename] / filename
        expected = spec["sha256"][filename]
        if destination.is_file() and not force:
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == expected:
                print(f"{name}: {destination.relative_to(ROOT)} verified")
                continue
            print(f"{name}: {destination.relative_to(ROOT)} has the wrong hash, re-downloading")
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = spec["base"] + filename
        print(f"{name}: downloading {url}")
        with urllib.request.urlopen(url, timeout=180) as response:
            destination.write_bytes(response.read())
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != expected:
            destination.unlink()
            raise SystemExit(f"{filename}: sha256 {digest} does not match the pinned {expected}")
        print(f"{name}: {destination.relative_to(ROOT)} ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="re-download even verified files")
    args = parser.parse_args()
    for name in (DATASETS if args.dataset == "all" else [args.dataset]):
        fetch(name, args.force)


if __name__ == "__main__":
    main()
