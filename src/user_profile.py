"""
user_profile.py — Per-user profile management and online weight learning.

Each user gets a profile.json stored at data/users/{username}/profile.json.
The profile stores:
  - preferences extracted by the Claude conversation
  - a 6-element learned weight vector updated after every thumbs-up/down
  - feedback history (for auditability)
  - conversation history (so Claude has context when the app reloads)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["genre", "energy", "valence", "danceability", "acousticness", "tempo"]

DEFAULT_WEIGHTS = np.array([0.35, 0.20, 0.20, 0.10, 0.10, 0.05], dtype=float)

LEARNING_RATE = 0.05

_ROOT = Path(__file__).parent.parent
_USERS_DIR = _ROOT / "data" / "users"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """Holds all mutable state for a single user session."""
    username: str
    preferences: Dict = field(default_factory=dict)
    weights: np.ndarray = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    feedback_history: List[Dict] = field(default_factory=list)
    conversation_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "username": self.username,
            "preferences": self.preferences,
            "weights": self.weights.tolist(),
            "feedback_history": self.feedback_history,
            "conversation_history": self.conversation_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(
            username=data["username"],
            preferences=data.get("preferences", {}),
            weights=np.array(data.get("weights", DEFAULT_WEIGHTS.tolist()), dtype=float),
            feedback_history=data.get("feedback_history", []),
            conversation_history=data.get("conversation_history", []),
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_profile(username: str) -> UserProfile:
    """Load a user profile from disk, or create a fresh one if not found."""
    path = _USERS_DIR / username / "profile.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return UserProfile.from_dict(json.load(f))
    return UserProfile(username=username)


def save_profile(profile: UserProfile) -> None:
    """Persist a user profile to disk."""
    path = _USERS_DIR / profile.username / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Online learning
# ---------------------------------------------------------------------------

def update_weights(profile: UserProfile, feature_scores: np.ndarray, liked: bool) -> None:
    """
    Perceptron-style online weight update.

    feature_scores: 6-element array in [0, 1] measuring per-feature match quality.
    liked: True if the user gave a thumbs-up, False for thumbs-down.

    Pushes weights toward features that mattered for liked songs, and away
    from features that dominated for disliked ones, then renormalizes.
    """
    signal = 1.0 if liked else -1.0
    profile.weights = profile.weights + LEARNING_RATE * signal * feature_scores
    profile.weights = np.clip(profile.weights, 0.01, None)
    profile.weights = profile.weights / profile.weights.sum()


def record_feedback(
    profile: UserProfile,
    song_id: int,
    song_title: str,
    feature_scores: np.ndarray,
    liked: bool,
) -> None:
    """Apply a weight update and append an entry to feedback_history."""
    update_weights(profile, feature_scores, liked)
    profile.feedback_history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "song_id": song_id,
        "song_title": song_title,
        "liked": liked,
        "feature_scores": feature_scores.tolist(),
        "weights_after": profile.weights.tolist(),
    })
