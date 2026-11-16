"""Cache-aware streaming ASR on top of the Transformers Nemotron RNN-T model."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

from muffled_asr.paths import ROOT

from . import adapter
from .stream import SampleBuffer

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
MODEL_DIR = ROOT / "models" / MODEL_ID.split("/")[-1]

# The `.nemo` and `.gguf` copies in the repo are for other runtimes; Transformers
# needs only these.
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


def is_downloaded(model_dir: Path = MODEL_DIR) -> bool:
    base = adapter.base_of(model_dir)
    return all((base / name).exists() for name in MODEL_FILES)


def ensure_model(model_dir: Path = MODEL_DIR) -> str:
    """Return a local path to the weights, fetching them into the project on first
    use so later runs never touch the network or the shared HF cache."""
    if not is_downloaded(model_dir):
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_ID, local_dir=model_dir, allow_patterns=MODEL_FILES)
    return str(model_dir)


def load_model(model_dir: Path, device: torch.device):
    """Processor and eval-mode model from a Transformers directory, or from an
    adapter directory on top of the base it points at."""
    path = ensure_model(adapter.base_of(model_dir))
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForRNNT.from_pretrained(path)
    if adapter.base_of(model_dir) != Path(model_dir):
        adapter.attach(model, model_dir)
    return processor, model.to(device).eval()


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class StreamingTranscriber:
    """Feeds fixed-size mel chunks from a `SampleBuffer` into a single long-lived
    `generate` call, which keeps the encoder cache warm across the whole session.
    """

    def __init__(self, device: torch.device, lookahead: int, model_dir: Path = MODEL_DIR) -> None:
        self.device = device
        self.processor, self.model = load_model(model_dir, device)
        self.processor.set_num_lookahead_tokens(lookahead)
        self.error: BaseException | None = None

    @property
    def lookahead(self) -> int:
        return self.processor.default_num_lookahead_tokens

    @property
    def latency_ms(self) -> int:
        """Delay the architecture imposes before a chunk can be emitted."""
        return self.processor.streaming_latency_ms

    @property
    def chunk_samples(self) -> int:
        """Audio consumed per streaming step. Lag naturally sawtooths up to about
        this much, so it is the yardstick for judging whether lag is a problem."""
        p = self.processor
        return p.num_mel_frames_per_audio_chunk * p.feature_extractor.hop_length

    def warmup(self) -> None:
        """Run one offline pass so lazily-built kernels are compiled before the
        first live chunk arrives."""
        fe = self.processor.feature_extractor
        silence = np.zeros(fe.sampling_rate, dtype=np.float32)
        inputs = self.processor(silence, sampling_rate=fe.sampling_rate, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        with torch.inference_mode():
            self.model.generate(**inputs)

    def start(self, buffer: SampleBuffer) -> TextIteratorStreamer:
        """Begin transcribing `buffer` on a worker thread.

        Returns the streamer to drain; it yields incremental text and finally
        `streamer.stop_signal` (None) when the session ends for any reason.
        """
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True)
        self.error = None
        threading.Thread(
            target=self._run, args=(buffer, streamer), name="asr", daemon=True
        ).start()
        return streamer

    def _run(self, buffer: SampleBuffer, streamer: TextIteratorStreamer) -> None:
        p = self.processor
        fe = p.feature_extractor
        try:
            first = buffer.read(0, p.num_samples_first_audio_chunk)
            if first is None:
                return
            first_inputs = p(
                first,
                sampling_rate=fe.sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=True,
                return_tensors="pt",
            ).to(self.device, dtype=self.model.dtype)

            def chunks():
                # The first chunk is the one already featurised above; every later
                # chunk is cut at `mel_frame * hop - n_fft // 2` so that its
                # uncentred STFT lands on exactly the frames a single centred pass
                # over the whole utterance would have produced.
                yield first_inputs.input_features[:, : p.num_mel_frames_first_audio_chunk, :]
                mel_frame = p.num_mel_frames_first_audio_chunk
                while True:
                    start = mel_frame * fe.hop_length - fe.n_fft // 2
                    block = buffer.read(start, p.num_samples_per_audio_chunk)
                    if block is None:
                        return
                    yield p(
                        block,
                        sampling_rate=fe.sampling_rate,
                        is_streaming=True,
                        is_first_audio_chunk=False,
                        return_tensors="pt",
                    ).input_features
                    mel_frame += p.num_mel_frames_per_audio_chunk

            # Built as a dict, not keywords: `first_inputs` already carries
            # `input_features` and `num_lookahead_tokens`, and the generator has to
            # replace the former rather than collide with it.
            kwargs = {**first_inputs, "input_features": chunks(), "streamer": streamer}
            with torch.inference_mode():
                self.model.generate(**kwargs)
        except BaseException as exc:  # surfaced in the GUI status bar
            self.error = exc
        finally:
            # `generate` signals the end itself on a clean exit; do it here too so
            # the GUI never waits forever on a failed or never-started session.
            streamer.text_queue.put(streamer.stop_signal)
