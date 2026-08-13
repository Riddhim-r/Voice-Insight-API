"""
Voice Insight API
------------------
Upload an audio file -> get back a cleaned transcript + structured summary.

Pipeline: preprocessing (denoise + VAD) -> transcription (Whisper) ->
summarization (Claude).

Run:
    uvicorn app.main:app --reload

Endpoints:
    GET  /health      -> liveness check
    POST /transcribe  -> preprocessing + transcription only
    POST /process     -> full pipeline: preprocessing + transcription + summarization
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.preprocessing import preprocess
from app.transcription import transcribe
from app.summarization import summarize

app = FastAPI(
    title="Voice Insight API",
    description="Noisy-audio transcription and summarization pipeline",
    version="0.1.0",
)

ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a", "audio/webm", "audio/ogg"
}


async def _read_and_validate(file: UploadFile) -> bytes:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return contents


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Voice Insight API"}


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """Preprocess + transcribe only. Useful for debugging the ASR stage in isolation."""
    contents = await _read_and_validate(file)

    try:
        prep = preprocess(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    try:
        result = transcribe(prep["samples"], prep["sample_rate"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")

    return JSONResponse({
        "transcript": result["text"],
        "language": result.get("language"),
        "segments": result.get("segments", []),
        "transcription_mode": result.get("mode"),
        "audio_diagnostics": {
            "original_duration_s": prep["original_duration_s"],
            "processed_duration_s": prep["processed_duration_s"],
            "silence_removed_s": prep["silence_removed_s"],
        },
    })


@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    """Full pipeline: preprocess -> transcribe -> summarize."""
    contents = await _read_and_validate(file)

    try:
        prep = preprocess(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    try:
        transcript_result = transcribe(prep["samples"], prep["sample_rate"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")

    try:
        summary_result = summarize(transcript_result["text"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {e}")

    return JSONResponse({
        "transcript": transcript_result["text"],
        "language": transcript_result.get("language"),
        "summary": summary_result.get("summary"),
        "key_points": summary_result.get("key_points", []),
        "action_items": summary_result.get("action_items", []),
        "topics": summary_result.get("topics", []),
        "audio_diagnostics": {
            "original_duration_s": prep["original_duration_s"],
            "processed_duration_s": prep["processed_duration_s"],
            "silence_removed_s": prep["silence_removed_s"],
        },
    })
