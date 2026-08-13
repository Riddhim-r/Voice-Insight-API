"""
Audio preprocessing pipeline.

Steps:
1. Load audio (any format pydub/ffmpeg supports)
2. Convert to mono, resample to 16kHz (standard for ASR models)
3. Apply spectral-gating noise reduction
4. Run Voice Activity Detection (VAD) to strip silence / non-speech segments

Returns a cleaned, speech-only 16kHz mono numpy array ready for feature
extraction / transcription along with audio diagnostics.
"""

import io
import struct
import numpy as np

# Ensure static-ffmpeg binary paths are loaded into environment if static_ffmpeg is installed
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

from pydub import AudioSegment
import noisereduce as nr
import webrtcvad


TARGET_SR = 16000


def load_audio(file_bytes: bytes) -> AudioSegment:
    """Load arbitrary audio bytes into a pydub AudioSegment."""
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    return audio


def to_mono_16k(audio: AudioSegment) -> AudioSegment:
    """Convert to mono, 16kHz, 16-bit PCM -- the standard ASR input format."""
    return audio.set_channels(1).set_frame_rate(TARGET_SR).set_sample_width(2)


def audiosegment_to_np(audio: AudioSegment) -> np.ndarray:
    """Convert pydub AudioSegment to a normalized float32 numpy array [-1.0, 1.0]."""
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # 16-bit PCM signed integer max is 32767
    max_val = float(np.iinfo(np.int16).max)
    if len(samples) > 0 and max_val > 0:
        samples /= max_val
    return samples


def np_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] numpy array back to 16-bit PCM bytes (for VAD)."""
    clipped = np.clip(samples, -1.0, 1.0)
    ints = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    return struct.pack("<%dh" % len(ints), *ints)


def denoise(samples: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Spectral-gating noise reduction. Estimates a noise profile from the
    signal itself and subtracts it -- effective as a fast first pass on
    steady background noise (fan hum, room tone, line noise).
    """
    if len(samples) == 0:
        return samples
    return nr.reduce_noise(y=samples, sr=sr, stationary=False)


def voice_activity_segments(samples: np.ndarray, sr: int = TARGET_SR,
                             frame_ms: int = 30, aggressiveness: int = 2):
    """
    Run WebRTC VAD over 30ms frames and return a boolean mask marking
    which frames contain speech. aggressiveness: 0 (least aggressive
    filtering) to 3 (most aggressive at cutting non-speech).
    """
    vad = webrtcvad.Vad(aggressiveness)
    pcm_bytes = np_to_pcm16_bytes(samples)

    frame_len = int(sr * frame_ms / 1000)  # samples per frame (480 for 30ms @ 16kHz)
    bytes_per_frame = frame_len * 2  # 16-bit = 2 bytes/sample (960 bytes)

    speech_mask = []
    for start in range(0, len(pcm_bytes) - bytes_per_frame + 1, bytes_per_frame):
        frame = pcm_bytes[start:start + bytes_per_frame]
        try:
            is_speech = vad.is_speech(frame, sr)
        except Exception:
            is_speech = True  # fallback: preserve frame on VAD exception
        speech_mask.append(is_speech)

    return speech_mask, frame_len


def strip_silence(samples: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Remove non-speech frames identified by VAD, concatenating speech-only audio."""
    if len(samples) == 0:
        return samples

    speech_mask, frame_len = voice_activity_segments(samples, sr)
    if not any(speech_mask):
        # Nothing detected as speech (e.g. silent/synthetic test tone) -- return as-is
        return samples

    kept = []
    for i, is_speech in enumerate(speech_mask):
        if is_speech:
            start = i * frame_len
            kept.append(samples[start:start + frame_len])
    return np.concatenate(kept) if kept else samples


def preprocess(file_bytes: bytes) -> dict:
    """
    Full preprocessing pipeline. Returns cleaned samples plus diagnostics
    useful for the API response / debugging.
    """
    raw_audio = load_audio(file_bytes)
    original_duration_s = len(raw_audio) / 1000.0

    normalized = to_mono_16k(raw_audio)
    samples = audiosegment_to_np(normalized)

    denoised = denoise(samples)
    speech_only = strip_silence(denoised)

    processed_duration_s = len(speech_only) / TARGET_SR if len(speech_only) > 0 else 0.0
    silence_removed_s = max(0.0, (len(denoised) / TARGET_SR) - processed_duration_s)

    return {
        "samples": speech_only,
        "sample_rate": TARGET_SR,
        "original_duration_s": round(original_duration_s, 2),
        "processed_duration_s": round(processed_duration_s, 2),
        "silence_removed_s": round(silence_removed_s, 2),
    }
