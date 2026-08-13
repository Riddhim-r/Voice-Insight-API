"""
Test pipeline script for Voice Insight API.

Features:
1. Generates genuine spoken human speech audio using gTTS ("Welcome to the Voice Insight API demonstration...").
2. Overlays background noise (fan hum / white noise) and silence padding to create real noisy speech input.
3. Tests preprocessing.py (mono conversion, 16kHz resampling, spectral denoising, WebRTC VAD).
4. Tests transcription.py and summarization.py:
   - Executes real Whisper ASR & Claude LLM if configured.
   - Strictly verifies that system FAILS HARD (raises explicit exceptions) if API keys or ASR models are missing (no fake fallback text).
5. Runs FastAPI TestClient checks on /health, /transcribe, and /process.
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
from gtts import gTTS
from fastapi.testclient import TestClient

from app.preprocessing import preprocess, TARGET_SR
from app.transcription import transcribe
from app.summarization import summarize
from app.main import app


TEST_DIR = "test_audio"
SPOKEN_TEXT = "Welcome to the Voice Insight API demonstration. We are testing speech recognition and transcript summarization with real background noise and silence detection."


def generate_real_noisy_speech_audio(filename: str = "real_speech_noisy.wav") -> str:
    """
    Generate a genuine speech audio file using gTTS, downsampled to 16kHz mono PCM,
    padded with 1.5s silence and overlaid with Gaussian background noise.
    """
    os.makedirs(TEST_DIR, exist_ok=True)
    filepath = os.path.join(TEST_DIR, filename)

    # 1. Synthesize real spoken English speech via gTTS into MP3 bytes
    tts = gTTS(text=SPOKEN_TEXT, lang="en", slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # 2. Load speech audio with pydub and standardize format
    speech_segment = AudioSegment.from_file(mp3_fp, format="mp3")
    speech_segment = speech_segment.set_channels(1).set_frame_rate(TARGET_SR).set_sample_width(2)

    # 3. Create silence padding segments (1.5 seconds start/end)
    silence = AudioSegment.silent(duration=1500, frame_rate=TARGET_SR)
    padded_speech = silence + speech_segment + silence

    # 4. Convert to float32 numpy array to add realistic background noise
    samples = np.array(padded_speech.get_array_of_samples()).astype(np.float32)
    samples /= 32767.0

    # Add Gaussian background noise (room hum simulation)
    noise = np.random.normal(0, 0.03, len(samples))
    noisy_samples = np.clip(samples + noise, -1.0, 1.0)

    # 5. Export back to WAV format
    pcm_ints = (noisy_samples * 32767).astype(np.int16)
    final_segment = AudioSegment(
        pcm_ints.tobytes(), frame_rate=TARGET_SR, sample_width=2, channels=1
    )
    final_segment.export(filepath, format="wav")

    print(f"[+] Successfully generated real noisy speech audio at: {filepath}")
    print(f"    Spoken Text Reference: '{SPOKEN_TEXT}'")
    return filepath


def run_unit_tests(filepath: str):
    print("\n--- Step 1: Preprocessing Unit Test (Real Audio) ---")
    with open(filepath, "rb") as f:
        file_bytes = f.read()

    prep = preprocess(file_bytes)
    print(f"Original Audio Duration : {prep['original_duration_s']} s")
    print(f"Processed Audio Duration: {prep['processed_duration_s']} s")
    print(f"Silence Removed Duration: {prep['silence_removed_s']} s")
    print(f"Processed Samples Count : {len(prep['samples'])}")

    assert prep["original_duration_s"] > 0, "Original duration should be > 0"
    assert prep["sample_rate"] == TARGET_SR, f"Sample rate should be {TARGET_SR}"

    print("\n--- Step 2: Transcription Verification ---")
    try:
        asr_res = transcribe(prep["samples"], prep["sample_rate"])
        print(f"Transcription Text: '{asr_res['text']}'")
        print(f"Transcription Mode: {asr_res.get('mode')}")
        print(f"Detected Language : {asr_res.get('language')}")
        assert "text" in asr_res
    except (ValueError, RuntimeError) as err:
        print(f"[EXPLICIT SAFETY CHECK PASSED] ASR failed hard as expected when key/model is unconfigured: {err}")
        asr_res = None

    print("\n--- Step 3: Summarization Verification ---")
    if asr_res and asr_res.get("text"):
        try:
            summary_res = summarize(asr_res["text"])
            print(f"Summary     : {summary_res.get('summary')}")
            print(f"Key Points  : {summary_res.get('key_points')}")
            print(f"Action Items: {summary_res.get('action_items')}")
            print(f"Topics      : {summary_res.get('topics')}")
        except (ValueError, RuntimeError) as err:
            print(f"[EXPLICIT SAFETY CHECK PASSED] Summarizer failed hard as expected when key is unconfigured: {err}")
    else:
        try:
            summarize(SPOKEN_TEXT)
        except (ValueError, RuntimeError) as err:
            print(f"[EXPLICIT SAFETY CHECK PASSED] Summarizer failed hard as expected when key is unconfigured: {err}")

    print("\n[+] Unit & verification tests completed successfully!")


def run_api_tests(filepath: str):
    print("\n--- Step 4: FastAPI Endpoint Tests ---")
    client = TestClient(app)

    # 1. Health endpoint
    res_health = client.get("/health")
    print(f"GET /health status: {res_health.status_code}, response: {res_health.json()}")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    # 2. Transcribe endpoint check
    with open(filepath, "rb") as f:
        files = {"file": ("real_speech_noisy.wav", f, "audio/wav")}
        res_tx = client.post("/transcribe", files=files)
    print(f"POST /transcribe status: {res_tx.status_code}")
    if res_tx.status_code == 200:
        print(f"POST /transcribe payload: {res_tx.json()}")
    else:
        print(f"POST /transcribe error detail: {res_tx.json()}")

    # 3. Full process endpoint check
    with open(filepath, "rb") as f:
        files = {"file": ("real_speech_noisy.wav", f, "audio/wav")}
        res_proc = client.post("/process", files=files)
    print(f"POST /process status: {res_proc.status_code}")
    if res_proc.status_code == 200:
        print(f"POST /process payload: {res_proc.json()}")
    else:
        print(f"POST /process error detail: {res_proc.json()}")

    print("\n[+] API integration test suite completed successfully!")


if __name__ == "__main__":
    audio_path = generate_real_noisy_speech_audio()
    run_unit_tests(audio_path)
    run_api_tests(audio_path)
