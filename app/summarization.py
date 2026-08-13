"""
Summarization module. Takes a raw transcript and produces a structured
summary + key points using an LLM, grounded strictly in the transcript
text (no outside knowledge injected).
"""

import os
import json
import logging
import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise meeting/audio summarizer. You will be
given a raw speech-to-text transcript, which may contain minor transcription
errors, filler words, or disfluencies. Summarize ONLY based on the content
in the transcript -- do not invent details that aren't present.

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "summary": "2-4 sentence high-level summary",
  "key_points": ["point 1", "point 2", "..."],
  "action_items": ["action 1", "..."],
  "topics": ["topic1", "topic2"]
}
If there are no clear action items, return an empty list for that field."""


def _fallback_summary(transcript_text: str) -> dict:
    """Fallback summarizer when ANTHROPIC_API_KEY is not set or call fails."""
    sentences = [s.strip() for s in transcript_text.split(".") if s.strip()]
    high_level = ". ".join(sentences[:2]) + "." if sentences else "Audio transcript recorded successfully."
    key_points = sentences[:3] if sentences else ["Audio processing completed."]
    
    return {
        "summary": f"Transcript summary: {high_level}",
        "key_points": key_points,
        "action_items": [],
        "topics": ["Speech Processing", "Audio Analysis"],
        "_mode": "fallback",
    }


def summarize(transcript_text: str) -> dict:
    if not transcript_text.strip():
        return {"summary": "", "key_points": [], "action_items": [], "topics": []}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set. Using fallback summarizer.")
        return _fallback_summary(transcript_text)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": transcript_text}],
        )

        raw_text = "".join(
            block.text for block in message.content if block.type == "text"
        )

        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            parsed["_mode"] = "anthropic_claude"
            return parsed
        except json.JSONDecodeError:
            return {
                "summary": raw_text,
                "key_points": [],
                "action_items": [],
                "topics": [],
                "_parse_warning": "Model did not return valid JSON; raw text returned in 'summary'.",
                "_mode": "anthropic_claude",
            }
    except Exception as e:
        logger.warning(f"Anthropic API call failed: {e}. Using fallback summarizer.")
        return _fallback_summary(transcript_text)
