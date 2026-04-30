"""
claude_client.py — Thin wrapper around the Anthropic API for preference elicitation.

Exposes three public functions:
  chat(messages)            — send a conversation turn and get a reply
  extract_preferences(text) — parse the <preferences>...</preferences> JSON block
  validate_genres(prefs)    — filter extracted genres to only those in the catalog
"""

import json
import os
import re
from typing import List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: Optional[anthropic.Anthropic] = None

# ---------------------------------------------------------------------------
# Load valid genres from catalog metadata so the system prompt stays in sync
# ---------------------------------------------------------------------------
_META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog_meta.json")


def _load_valid_genres() -> List[str]:
    try:
        with open(_META_PATH) as f:
            return json.load(f)["genres"]
    except Exception:
        return []


VALID_GENRES: List[str] = _load_valid_genres()
_GENRES_STR = ", ".join(VALID_GENRES)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""\
You are a friendly music taste assistant helping users discover songs they'll love.

GUARDRAILS — follow these at all times:
1. You only discuss topics related to music: songs, artists, albums, genres, moods, and recommendations.
   If the user asks about anything unrelated to music (e.g. sports, cooking, coding, news, etc.),
   politely decline and redirect them back to music discovery.
2. When the user mentions a genre, it must come from this approved catalog list:
   {_GENRES_STR}
   If they request a genre not on this list, kindly explain that it isn't in our catalog and
   suggest the closest available options from the list above. Never include an unsupported genre
   in the <preferences> block.

Your job is to have a short, natural conversation to understand the user's music preferences.
Ask about:
- Favorite genres (choose from the approved list above)
- Energy level they want right now (chill/relaxed 0.0–0.3 vs. mid 0.4–0.6 vs. high-energy 0.7–1.0)
- Whether they prefer acoustic or electronic/produced sounds
- The mood/vibe they want (happy and upbeat = high valence ~0.8; mellow or bittersweet = mid ~0.5; melancholy = low ~0.3)
- Tempo preference (slow ~0.2, medium ~0.5, fast ~0.8 on a 0–1 scale)

Keep it conversational — ask one or two things at a time. After 3–4 exchanges, emit EXACTLY this block:

<preferences>
{{
  "favorite_genres": ["genre1", "genre2"],
  "target_energy": 0.0,
  "target_valence": 0.0,
  "target_danceability": 0.0,
  "likes_acoustic": true,
  "target_tempo": 0.0
}}
</preferences>

Only include genres from the approved catalog list in favorite_genres.
Then say something warm like "Great, pulling up your recommendations now!"
"""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def chat(messages: List[dict], max_tokens: int = 1024) -> str:
    """
    Send a conversation turn to Claude and return the assistant reply text.

    messages: list of {"role": "user"/"assistant", "content": str}
    """
    response = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def extract_preferences(text: str) -> Optional[dict]:
    """
    Parse the <preferences>…</preferences> JSON block from a Claude reply.
    Returns None if the block is absent or malformed.
    """
    match = re.search(r"<preferences>(.*?)</preferences>", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def validate_genres(prefs: dict) -> Tuple[dict, List[str]]:
    """
    Programmatic safety net: remove any favorite_genres not in VALID_GENRES.

    Returns (cleaned_prefs, rejected_genres). cleaned_prefs is a shallow copy
    with only catalog-valid genres; rejected_genres lists what was stripped out.
    """
    raw = prefs.get("favorite_genres", [])
    valid = [g for g in raw if g in VALID_GENRES]
    rejected = [g for g in raw if g not in VALID_GENRES]
    cleaned = {**prefs, "favorite_genres": valid}
    return cleaned, rejected
