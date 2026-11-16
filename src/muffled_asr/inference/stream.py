"""Microphone capture and the buffers the GUI and the ASR engine read from."""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000


class SampleBuffer:
    """Append-only audio addressed by absolute sample index, safe across threads.

    The ASR feature extractor asks for overlapping windows, so it addresses audio
    by absolute position rather than popping chunks. Every read drops the samples
    before its own start, which is the oldest position any later read can ask for,
    so memory stays bounded by how far the reader lags the microphone.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._data = np.zeros(0, dtype=np.float32)
        self._start = 0  # absolute index of _data[0]
        self._written = 0  # samples captured so far
        self._consumed = 0  # absolute end of the last completed read
        self._closed = False

    def write(self, block: np.ndarray) -> None:
        with self._cv:
            if self._closed:
                return
            self._data = np.concatenate((self._data, block))
            self._written += len(block)
            self._cv.notify_all()

    def read(self, start: int, count: int) -> np.ndarray | None:
        """Block until ``[start, start + count)`` has been captured and return it.

        A negative `start` reads as silence, matching the zero-padding a centred
        STFT assumes before the signal begins; the shortest chunk setting asks for
        exactly that. Returns ``None`` if the buffer is closed before the range is
        complete, which is how the caller learns the stream has ended.
        """
        lead = max(0, -start)
        start += lead
        count -= lead
        with self._cv:
            while not self._closed and self._start + len(self._data) < start + count:
                self._cv.wait()
            if self._start + len(self._data) < start + count:
                return None
            offset = start - self._start
            chunk = self._data[offset : offset + count].copy()
            self._data = self._data[offset:]
            self._start = start
            self._consumed = start + count
        if lead:
            chunk = np.concatenate((np.zeros(lead, dtype=np.float32), chunk))
        return chunk

    def lag(self) -> int:
        """Samples captured since the model last finished reading a chunk.

        This oscillates between roughly zero and one chunk while the model keeps
        up, and climbs without bound when it does not.
        """
        with self._cv:
            return self._written - self._consumed

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class Waveform:
    """Ring buffer holding the most recent samples, for display only."""

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._lock = threading.Lock()

    def write(self, block: np.ndarray) -> None:
        with self._lock:
            n = len(block)
            if n >= len(self._buf):
                self._buf[:] = block[-len(self._buf) :]
            else:
                self._buf[:-n] = self._buf[n:]
                self._buf[-n:] = block

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._buf.copy()

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0.0


class Microphone:
    """16 kHz mono capture, fanned out to every buffer given to it."""

    def __init__(self, sinks, device=None, blocksize: int = 1024) -> None:
        self._sinks = sinks
        self.overflows = 0
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=blocksize,
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status.input_overflow:
            self.overflows += 1
        block = indata[:, 0].copy()
        for sink in self._sinks:
            sink.write(block)

    def __enter__(self) -> "Microphone":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()


def describe_default_input(device=None) -> str:
    info = sd.query_devices(device, "input")
    return f"{info['name']} ({info['max_input_channels']} ch, {int(info['default_samplerate'])} Hz native)"


def list_input_devices() -> str:
    lines = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            lines.append(f"  [{idx}] {dev['name']}  ({dev['max_input_channels']} ch)")
    return "\n".join(lines) or "  (no input devices found)"
