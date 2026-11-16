"""Tkinter front end: live waveform on top, read-only transcript below."""

from __future__ import annotations

import collections
import queue
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from .asr import MODEL_DIR, StreamingTranscriber, is_downloaded
from .stream import SAMPLE_RATE, Microphone, SampleBuffer, Waveform, describe_default_input

WAVEFORM_SECONDS = 4.0
REFRESH_MS = 33
SPEECH_THRESHOLD_DBFS = -42.0
LEVEL_WINDOW = SAMPLE_RATE // 10  # 100 ms

BG = "#11141a"
PANEL = "#1a1f28"
TRACE = "#4ade80"
AXIS = "#2b3341"
MUTED = "#8b98ab"
TEXT = "#e6edf3"
ALERT = "#f87171"


def dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return 20.0 * np.log10(max(rms, 1e-7))


class App(tk.Tk):
    def __init__(self, device_index, torch_device, lookahead: int, model_dir=MODEL_DIR) -> None:
        super().__init__()
        self.title("Nemotron Streaming STT")
        self.geometry("1000x680")
        self.minsize(720, 520)
        self.configure(bg=BG)

        self._device_index = device_index
        self._torch_device = torch_device
        self._initial_lookahead = lookahead
        self._model_dir = model_dir

        self.transcriber: StreamingTranscriber | None = None
        self.mic: Microphone | None = None
        self.buffer: SampleBuffer | None = None
        self.streamer = None
        self.waveform = Waveform(int(SAMPLE_RATE * WAVEFORM_SECONDS))
        self._lag_history = collections.deque(maxlen=int(1000 / REFRESH_MS))
        self._load_result: queue.Queue = queue.Queue()

        self._build_ui()
        self._set_status(
            f"Loading model on {torch_device.type.upper()}..."
            if is_downloaded(model_dir)
            else "Downloading model (~2.4 GB) into models/ - progress in terminal",
            MUTED,
        )
        threading.Thread(target=self._load_model, name="load", daemon=True).start()
        self.after(REFRESH_MS, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=MUTED, font=("Helvetica", 12))
        style.configure("Value.TLabel", foreground=TEXT)
        style.configure("TButton", font=("Helvetica", 12))

        bar = ttk.Frame(self, padding=(14, 12, 14, 8))
        bar.pack(fill="x")

        self.start_btn = ttk.Button(bar, text="Start", command=self.start, state="disabled")
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Clear", command=self._clear_text).pack(side="left", padx=(8, 0))

        ttk.Label(bar, text="Latency").pack(side="left", padx=(24, 6))
        self.latency_box = ttk.Combobox(bar, width=10, state="disabled", values=[])
        self.latency_box.pack(side="left")
        self.latency_box.bind("<<ComboboxSelected>>", self._on_latency_change)

        self.status = ttk.Label(bar, text="", style="Value.TLabel")
        self.status.pack(side="right")

        ttk.Label(
            self, text=f"Input: {describe_default_input(self._device_index)}", padding=(16, 0)
        ).pack(fill="x")

        self.canvas = tk.Canvas(
            self, bg=PANEL, highlightthickness=1, highlightbackground=AXIS, height=220
        )
        self.canvas.pack(fill="both", expand=False, padx=14, pady=(8, 0))
        self.canvas.bind("<Configure>", lambda _e: self._draw_waveform())
        self._axis = self.canvas.create_line(0, 0, 0, 0, fill=AXIS)
        self._trace = self.canvas.create_line(0, 0, 0, 0, fill=TRACE, width=1)

        meters = ttk.Frame(self, padding=(14, 10, 14, 4))
        meters.pack(fill="x")
        self.speech_dot = tk.Label(
            meters, text="●  idle", bg=BG, fg=MUTED, font=("Helvetica", 12, "bold")
        )
        self.speech_dot.pack(side="left")
        self.level = ttk.Progressbar(meters, length=220, maximum=60.0)
        self.level.pack(side="left", padx=(16, 10))
        self.level_label = ttk.Label(meters, text="-inf dBFS", width=12)
        self.level_label.pack(side="left")
        self.lag_label = ttk.Label(meters, text="")
        self.lag_label.pack(side="right")

        wrap = ttk.Frame(self, padding=(14, 4, 14, 14))
        wrap.pack(fill="both", expand=True)
        self.text = tk.Text(
            wrap,
            wrap="word",
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            padx=12,
            pady=10,
            font=("Helvetica", 15),
            state="disabled",
            highlightthickness=1,
            highlightbackground=AXIS,
        )
        scroll = ttk.Scrollbar(wrap, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

    def _set_status(self, message: str, colour: str = TEXT) -> None:
        self.status.configure(text=message, foreground=colour)

    def _clear_text(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def _append_text(self, chunk: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", chunk)
        self.text.see("end")
        self.text.configure(state="disabled")

    # ------------------------------------------------------------- model

    def _load_model(self) -> None:
        try:
            engine = StreamingTranscriber(
                self._torch_device, self._initial_lookahead, self._model_dir
            )
            engine.warmup()
            self._load_result.put(engine)
        except BaseException as exc:
            self._load_result.put(exc)

    def _on_model_ready(self, engine: StreamingTranscriber) -> None:
        self.transcriber = engine
        latencies = engine.processor.supported_streaming_latencies_ms
        self._latency_to_lookahead = {f"{ms} ms": la for la, ms in latencies.items()}
        self.latency_box.configure(
            values=sorted(self._latency_to_lookahead, key=lambda s: int(s.split()[0])),
            state="readonly",
        )
        self.latency_box.set(f"{engine.latency_ms} ms")
        self.start_btn.configure(state="normal")
        self._set_status(f"Ready on {engine.device.type.upper()}", TRACE)

    def _on_latency_change(self, _event) -> None:
        if self.transcriber is None:
            return
        self.transcriber.processor.set_num_lookahead_tokens(
            self._latency_to_lookahead[self.latency_box.get()]
        )

    # ----------------------------------------------------------- session

    def start(self) -> None:
        if self.transcriber is None or self.mic is not None:
            return
        self.waveform.clear()
        self._lag_history.clear()
        self.buffer = SampleBuffer()
        try:
            self.mic = Microphone([self.buffer, self.waveform], device=self._device_index)
            self.mic.__enter__()
        except BaseException as exc:
            self.mic = None
            self.buffer = None
            self._set_status(f"Microphone error: {exc}", ALERT)
            return
        # Started only once the microphone is live, so a capture failure cannot be
        # mistaken for a finished session.
        self.streamer = self.transcriber.start(self.buffer)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.latency_box.configure(state="disabled")
        self._set_status(f"Listening - {self.transcriber.latency_ms} ms latency", TRACE)

    def stop(self) -> None:
        if self.mic is None:
            return
        self.mic.__exit__()
        self.mic = None
        self.buffer.close()  # unblocks the ASR worker, which then ends the stream
        self.stop_btn.configure(state="disabled")
        self._set_status("Finishing...", MUTED)

    def _on_session_end(self) -> None:
        error = self.transcriber.error if self.transcriber else None
        self.streamer = None
        self.buffer = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.latency_box.configure(state="readonly")
        self.lag_label.configure(text="")
        if error is not None:
            self._set_status(f"{type(error).__name__}: {error}", ALERT)
        else:
            self._set_status("Stopped", MUTED)

    # -------------------------------------------------------------- loop

    def _tick(self) -> None:
        self._draw_waveform()
        self._drain_text()
        self.after(REFRESH_MS, self._tick)

    def _drain_text(self) -> None:
        if not self._load_result.empty():
            result = self._load_result.get()
            if isinstance(result, BaseException):
                self._set_status(f"Model load failed: {result}", ALERT)
            else:
                self._on_model_ready(result)

        if self.streamer is None:
            return
        while True:
            try:
                chunk = self.streamer.text_queue.get_nowait()
            except queue.Empty:
                break
            if chunk is self.streamer.stop_signal:
                self._on_session_end()
                return
            self._append_text(chunk)

        if self.buffer is not None and self.transcriber is not None:
            # Lag sawtooths from ~0 to one chunk as each chunk fills, so show the
            # peak over the last second; a value beyond a chunk plus a second means
            # the model is genuinely falling behind.
            self._lag_history.append(self.buffer.lag())
            lag = max(self._lag_history)
            behind = self.transcriber.chunk_samples + SAMPLE_RATE
            self.lag_label.configure(
                text=f"lag {lag / SAMPLE_RATE * 1000:.0f} ms",
                foreground=ALERT if lag > behind else MUTED,
            )

    def _draw_waveform(self) -> None:
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        mid = height / 2
        self.canvas.coords(self._axis, 0, mid, width, mid)

        samples = self.waveform.snapshot()
        columns = min(width, len(samples))
        usable = len(samples) - len(samples) % columns
        blocks = samples[:usable].reshape(columns, -1)
        lo = np.clip(blocks.min(axis=1), -1.0, 1.0)
        hi = np.clip(blocks.max(axis=1), -1.0, 1.0)

        # One polyline alternating each column's min and max reads as a filled
        # envelope while staying a single canvas item to update per frame.
        xs = np.linspace(0, width, columns)
        points = np.empty(columns * 4)
        points[0::4] = xs
        points[1::4] = mid - hi * mid * 0.92
        points[2::4] = xs
        points[3::4] = mid - lo * mid * 0.92
        self.canvas.coords(self._trace, *points.tolist())

        level = dbfs(samples[-LEVEL_WINDOW:])
        self.level.configure(value=max(0.0, level + 60.0))
        self.level_label.configure(text="  -inf dBFS" if level < -99 else f"{level:6.1f} dBFS")
        speaking = self.mic is not None and level > SPEECH_THRESHOLD_DBFS
        self.speech_dot.configure(
            text="●  SPEECH" if speaking else "●  idle",
            fg=TRACE if speaking else MUTED,
        )

    def _on_close(self) -> None:
        if self.mic is not None:
            self.mic.__exit__()
            self.buffer.close()
        self.destroy()
