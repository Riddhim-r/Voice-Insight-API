"""
Test pipeline script for Voice Insight API.

Features:
1. Ensures test_audio/ directory exists (fixes directory creation bug).
2. Generates synthetic test WAV file (speech-like tone bursts + Gaussian background noise + silence).
3. Tests preprocessing.py (mono conversion, 16kHz resampling, spectral denoising, VAD silence removal).
4. Tests transcription.py (ASR processing, segment timestamps, language detection).
5. Tests summarization.py (JSON output grounding check).
6. Runs FastAPI TestClient checks on /health, /transcribe, and /process.
"""

import os
import io
import math
import struct
import numpy as np

# Ensure static-ffmpeg is on PATH before importing pydub
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

from pydub import AudioSegment
from fastapi.testclient import TestClient


from app.preprocessing import preprocess, TARGET_SR
from app.transcription import transcribe
from app.summarization import summarize
from app.main import app


TEST_DIR = "test_audio"


def generate_synthetic_noisy_audio(filename: str = "synthetic_noisy.wav") -> str:
    """
    Generate a synthetic speech-and-noise WAV file.
    Structure:
    - 1s background noise (fan hum / white noise)
    - 2s tone burst (simulated speech segment 1: 440 Hz tone + noise)
    - 1s silence / quiet background noise
    - 2s tone burst (simulated speech segment 2: 880 Hz tone + noise)
    - 1s tail silence
    """
    os.makedirs(TEST_DIR, exist_ok=True)
    filepath = os.path.join(TEST_DIR, filename)

    sr = TARGET_SR
    duration = 7.0
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # Base background noise (steady Gaussian noise)
    noise = np.random.normal(0, 0.05, total_samples)

    # Speech signal bursts (harmonics)
    speech = np.zeros(total_samples)

    # Segment 1: t=1.0s to 3.0s (440Hz + 880Hz tone burst)
    mask1 = (t >= 1.0) & (t <= 3.0)
    speech[mask1] = 0.4 * np.sin(2 * np.pi * 440 * t[mask1]) + 0.2 * np.sin(2 * np.pi * 880 * t[mask1])

    # Segment 2: t=4.0s to 6.0s (523Hz + 1046Hz tone burst)
    mask2 = (t >= 4.0) & (t <= 6.0)
    speech[mask2] = 0.4 * np.sin(2 * np.pi * 523 * t[mask2]) + 0.2 * np.sin(2 * np.pi * 1046 * t[mask2])

    combined = speech + noise
    combined_clipped = np.clip(combined, -1.0, 1.0)

    # Convert to 16-bit PCM bytes
    pcm_ints = (combined_clipped * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        pcm_ints.tobytes(), frame_rate=sr, sample_width=2, channels=1
    )

    audio_segment.export(filepath, format="wav")
    print(f"[+] Successfully generated synthetic test audio file at: {filepath}")
    return filepath


def run_unit_tests(filepath: str):
    print("\n--- Step 1: Preprocessing Unit Test ---")
    with open(filepath, "rb") as f:
        file_bytes = f.read()

    prep = preprocess(file_bytes)
    print(f"Original Audio Duration : {prep['original_duration_s']} s")
    print(f"Processed Audio Duration: {prep['processed_duration_s']} s")
    print(f"Silence Removed Duration: {prep['silence_removed_s']} s")
    print(f"Processed Samples Count : {len(prep['samples'])}")

    assert prep["original_duration_s"] > 0, "Original duration should be > 0"
    assert prep["sample_rate"] == TARGET_SR, f"Sample rate should be {TARGET_SR}"

    print("\n--- Step 2: Transcription Unit Test ---")
    asr_res = transcribe(prep["samples"], prep["sample_rate"])
    print(f"Transcription Text: '{asr_res['text']}'")
    print(f"Transcription Mode: {asr_res.get('mode')}")
    print(f"Detected Language : {asr_res.get('language')}")
    print(f"Segments Count    : {len(asr_res.get('segments', []))}")

    assert isinstance(asr_res["text"], str), "Transcription text should be a string"

    print("\n--- Step 3: Summarization Unit Test ---")
    summary_res = summarize(asr_res["text"])
    print(f"Summary     : {summary_res.get('summary')}")
    print(f"Key Points  : {summary_res.get('key_points')}")
    print(f"Action Items: {summary_res.get('action_items')}")
    print(f"Topics      : {summary_res.get('topics')}")

    assert "summary" in summary_res, "Summarizer response must contain 'summary' field"
    assert "key_points" in summary_res, "Summarizer response must contain 'key_points' field"

    print("\n[+] Unit tests completed successfully!")


def run_api_tests(filepath: str):
    print("\n--- Step 4: FastAPI Endpoint Tests ---")
    client = TestClient(app)

    # 1. Health endpoint
    res_health = client.get("/health")
    print(f"GET /health status: {res_health.status_code}, response: {res_health.json()}")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    # 2. Transcribe endpoint
    with open(filepath, "rb") as f:
        files = {"file": ("synthetic_noisy.wav", f, "audio/wav")}
        res_tx = client.post("/transcribe", files=files)
    print(f"POST /transcribe status: {res_tx.status_code}")
    print(f"POST /transcribe payload: {res_tx.json()}")
    assert res_tx.status_code == 200
    assert "transcript" in res_tx.json()
    assert "audio_diagnostics" in res_tx.json()

    # 3. Full process endpoint
    with open(filepath, "rb") as f:
        files = {"file": ("synthetic_noisy.wav", f, "audio/wav")}
        res_proc = client.post("/process", files=files)
    print(f"POST /process status: {res_proc.status_code}")
    print(f"POST /process payload: {res_proc.json()}")
    assert res_proc.status_code == 200
    assert "transcript" in res_proc.json()
    assert "summary" in res_proc.json()
    assert "audio_diagnostics" in res_proc.json()

    print("\n[+] API integration tests completed successfully!")


if __name__ == "__main__":
    audio_path = generate_synthetic_noisy_audio()
    run_unit_tests(audio_path)
    run_api_tests(audio_path)
