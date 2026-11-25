"""Transcript loading: YouTube API first, manual/uploaded fallback."""
import json
import re
from typing import List, Dict

Segment = Dict[str, float]  # {"text": str, "start": float, "duration": float}


def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL or a raw 11-char video id."""
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def fetch_from_youtube(video_id: str, languages=("en",)) -> List[Segment]:
    """Try the official transcript API. Raises on failure (blocked, disabled, etc.)."""
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]


def load_manual(path_or_file) -> List[Segment]:
    """Load a manual transcript from JSON: [{"text","start","duration"}, ...]."""
    if hasattr(path_or_file, "read"):
        data = json.load(path_or_file)
    else:
        with open(path_or_file, encoding="utf-8") as f:
            data = json.load(f)
    return [{"text": d["text"], "start": float(d["start"]),
             "duration": float(d.get("duration", 0))} for d in data]


def get_transcript(video_id: str, fallback_path=None) -> tuple[List[Segment], str]:
    """Returns (segments, source). Falls back to a manual transcript when YouTube blocks access."""
    try:
        return fetch_from_youtube(video_id), "youtube_api"
    except Exception as err:  # TranscriptsDisabled, IP blocks, network errors...
        if fallback_path is None:
            raise RuntimeError(f"YouTube transcript unavailable ({err}). Upload a manual transcript.")
        return load_manual(fallback_path), "manual_fallback"


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
