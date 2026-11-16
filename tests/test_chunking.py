"""Does the live chunk slicing reproduce a single centred offline pass?"""
import numpy as np, torch
from transformers import AutoProcessor
from muffled_asr.inference.asr import ensure_model
from muffled_asr.inference.stream import SampleBuffer

p = AutoProcessor.from_pretrained(ensure_model())
fe = p.feature_extractor
rng = np.random.default_rng(0)
audio = (rng.standard_normal(16000 * 5) * 0.1).astype(np.float32)

offline = p(audio, sampling_rate=fe.sampling_rate, return_tensors="pt").input_features

for lookahead in p.supported_num_lookahead_tokens:
    p.set_num_lookahead_tokens(lookahead)
    buf = SampleBuffer()
    buf.write(audio)
    buf.close()

    first = buf.read(0, p.num_samples_first_audio_chunk)
    assert first is not None
    out = [p(first, sampling_rate=fe.sampling_rate, is_streaming=True,
             is_first_audio_chunk=True, return_tensors="pt"
             ).input_features[:, : p.num_mel_frames_first_audio_chunk, :]]
    mel = p.num_mel_frames_first_audio_chunk
    while True:
        start = mel * fe.hop_length - fe.n_fft // 2
        block = buf.read(start, p.num_samples_per_audio_chunk)
        if block is None:
            break
        out.append(p(block, sampling_rate=fe.sampling_rate, is_streaming=True,
                     is_first_audio_chunk=False, return_tensors="pt").input_features)
        mel += p.num_mel_frames_per_audio_chunk

    streamed = torch.cat(out, dim=1)
    n = streamed.shape[1]
    assert out[0].shape[1] == p.num_mel_frames_first_audio_chunk, out[0].shape
    assert all(c.shape[1] == p.num_mel_frames_per_audio_chunk for c in out[1:])
    diff = (streamed - offline[:, :n, :]).abs().max().item()
    stride_ms = p.num_mel_frames_per_audio_chunk * fe.hop_length / fe.sampling_rate * 1000
    print(f"lookahead={lookahead:2d}  latency={p.streaming_latency_ms:5d}ms  "
          f"stride={stride_ms:7.1f}ms  frames={n:4d}/{offline.shape[1]}  maxdiff={diff:.2e}")
    assert diff < 1e-3, f"chunking diverges from offline pass: {diff}"

print("\nOK: streaming chunks match the offline features for every latency setting.")
