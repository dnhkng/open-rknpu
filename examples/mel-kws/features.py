"""SPDX-License-Identifier: MIT
Audio front end for the mel-CNN spoken-digit example: WAV -> 3x32x32 UINT8-ready image.

The NPU input is a single UINT8 tensor, so all three channels must share one affine
range. Features are therefore fixed to `[0, 1]` by construction:

* channel 0 - log-mel, per-utterance mean/std normalized and clipped to +-3
  (`(x + 3) / 6`);
* channel 1 - first difference of the normalized log-mel, clipped to +-3;
* channel 2 - second difference, clipped to +-3.

The runtime then maps `byte = round(value * 255)` with input scale `1/255`, zero point 0.
The transform is deterministic and dependency-free (stdlib `wave` + NumPy), so a board
input file can be regenerated from the source WAV byte for byte.
"""
from pathlib import Path
import wave

import numpy as np

SAMPLE_RATE = 8000
SAMPLES = 4224          # 32 frames of 128 samples + one 256-sample window (528 ms)
WINDOW = 256
HOP = 128
FFT = 256
MELS = 32
FMIN, FMAX = 20.0, 4000.0
CLIP = 3.0


def read_wav(path):
    """Mono float32 samples in [-1, 1] at 8 kHz."""
    with wave.open(str(path)) as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 8 kHz mono 16-bit PCM")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def fix_length(samples, length=SAMPLES):
    """Center-crop long recordings, zero-pad short ones symmetrically."""
    if samples.size >= length:
        start = (samples.size - length) // 2
        return samples[start:start + length]
    pad = length - samples.size
    left = pad // 2
    return np.pad(samples, (left, pad - left))


def mel_filterbank(mels=MELS, fft=FFT, rate=SAMPLE_RATE, low=FMIN, high=FMAX):
    """Triangular mel filters, each normalized to unit sum (HTK mel scale)."""
    def to_mel(frequency):
        return 2595.0 * np.log10(1.0 + frequency / 700.0)

    def to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    edges = to_hz(np.linspace(to_mel(low), to_mel(high), mels + 2))
    bins = np.floor((fft + 1) * edges / rate).astype(int)
    filters = np.zeros((mels, fft // 2 + 1), dtype=np.float32)
    for index in range(mels):
        left, center, right = bins[index], bins[index + 1], bins[index + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for bin_index in range(left, min(center, filters.shape[1])):
            filters[index, bin_index] = (bin_index - left) / (center - left)
        for bin_index in range(center, min(right, filters.shape[1])):
            filters[index, bin_index] = (right - bin_index) / (right - center)
    return filters / np.maximum(filters.sum(axis=1, keepdims=True), 1e-9)


def log_mel(samples, filters=None):
    """32x32 per-utterance normalized log-mel image."""
    filters = mel_filterbank() if filters is None else filters
    frames = 1 + (samples.size - WINDOW) // HOP
    window = np.hanning(WINDOW).astype(np.float32)
    strided = np.lib.stride_tricks.sliding_window_view(samples, WINDOW)[::HOP][:frames]
    spectrum = np.fft.rfft(strided * window, n=FFT, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    energies = power @ filters.T
    image = np.log(energies + 1e-6)
    image = (image - image.mean()) / max(float(image.std()), 1e-5)
    return image.astype(np.float32)


def diffs(image):
    """First and second differences along time, with edge replication."""
    padded = np.pad(image, ((0, 0), (1, 1)), mode="edge")
    first = (padded[:, 2:] - padded[:, :-2]) / 2.0
    padded_first = np.pad(first, ((0, 0), (1, 1)), mode="edge")
    second = (padded_first[:, 2:] - padded_first[:, :-2]) / 2.0
    return first, second


def features(path, filters=None):
    """3x32x32 float32 image in [0, 1] ready for UINT8 packing at scale 1/255.

    Every channel is standardized on its own (log-mel over the whole utterance, the two
    difference channels over the whole image) before the +-3 clip, so all three channels
    use the full 8-bit range instead of the deltas collapsing into a few codes.
    """
    image = log_mel(fix_length(read_wav(path)), filters)
    first, second = diffs(image)
    channels = [image]
    for difference in (first, second):
        channels.append(difference / max(float(difference.std()), 1e-5))
    stacked = np.stack(channels)
    return ((np.clip(stacked, -CLIP, CLIP) + CLIP) / (2 * CLIP)).astype(np.float32)


def to_uint8(image):
    """The exact bytes the runtime expects: round(value * 255), clipped."""
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def split(recordings):
    """The dataset's official split: recording index 0-4 test, 5-49 train."""
    test, train = [], []
    for path in sorted(Path(recordings).glob("*.wav")):
        digit, speaker, index = path.stem.split("_")
        entry = (path, int(digit))
        (test if int(index) < 5 else train).append(entry)
    return train, test
