"""SPDX-License-Identifier: MIT
Fetch the Free Spoken Digit Dataset (FSDD) used by the mel-CNN example.

The dataset is **not** vendored: it is Creative Commons Attribution-ShareAlike 4.0
(https://creativecommons.org/licenses/by-sa/4.0/), so this script downloads the pinned
upstream archive, verifies its sha256 and extracts the 3,000 recordings under
`research/pretrained/fsdd/`. Recordings are named `{digit}_{speaker}_{index}.wav`, and the
dataset's official split is index 0-4 test / 5-49 train.

    PYTHONPATH=src python examples/mel-kws/fetch_data.py

Provenance (URL, archive sha256, license) is written to `source.json` next to the data.
"""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "research/pretrained/fsdd"
ARCHIVE = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
SHA256 = "63938a0ef8dee8870149d18fd455ff86e5edb1c7a783b87dd39f23a730cb6d53"
LICENSE = "CC BY-SA 4.0"


def main():
    recordings = DEST / "recordings"
    if recordings.is_dir() and len(list(recordings.glob("*.wav"))) == 3000:
        print(f"already present: {len(list(recordings.glob('*.wav')))} recordings in {recordings}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    cached = Path("/tmp/fsdd.zip")
    if not cached.is_file():
        print(f"downloading {ARCHIVE}")
        with urllib.request.urlopen(ARCHIVE, timeout=120) as response, cached.open("wb") as out:
            shutil.copyfileobj(response, out)
    digest = hashlib.sha256(cached.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"archive sha256 {digest} does not match the pinned {SHA256}")
    with zipfile.ZipFile(cached) as archive:
        names = [n for n in archive.namelist() if n.endswith(".wav") and "/recordings/" in n]
        if len(names) != 3000:
            raise SystemExit(f"expected 3000 recordings, found {len(names)}")
        recordings.mkdir(parents=True, exist_ok=True)
        for name in names:
            with archive.open(name) as source, (recordings / Path(name).name).open("wb") as target:
                shutil.copyfileobj(source, target)
    (DEST / "source.json").write_text(json.dumps({
        "dataset": "Free Spoken Digit Dataset (FSDD)",
        "url": ARCHIVE,
        "archive_sha256": SHA256,
        "license": LICENSE,
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "recordings": 3000,
        "speakers": 6,
        "sample_rate": 8000,
        "split": "index 0-4 test, 5-49 train (official)",
        "note": "Data is not redistributed with this repository; this script downloads it.",
    }, indent=2) + "\n")
    print(f"extracted {len(names)} recordings into {recordings}")


if __name__ == "__main__":
    main()
