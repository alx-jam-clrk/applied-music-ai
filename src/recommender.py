"""
recommender.py — Content-based music recommender with learnable per-user weights.

Scoring uses 6 features: genre, energy, valence, danceability, acousticness, tempo.
Weights live in UserProfile and shift after every thumbs-up / thumbs-down.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.catalog import get_tempo_bounds
from src.user_profile import FEATURE_NAMES, UserProfile


# ---------------------------------------------------------------------------
# Feature scoring
# ---------------------------------------------------------------------------

def compute_feature_scores(
    song: pd.Series,
    prefs: dict,
    tempo_min: float,
    tempo_max: float,
) -> np.ndarray:
    """
    Return a 6-element array of per-feature match scores in [0, 1].

    Indices map to FEATURE_NAMES: genre, energy, valence,
    danceability, acousticness, tempo.
    """
    scores = np.zeros(6, dtype=float)

    # genre: 1.0 if song genre is in the user's preferred genres list
    preferred = prefs.get("favorite_genres", [])
    if isinstance(preferred, str):
        preferred = [preferred]
    scores[0] = 1.0 if song["genre"] in preferred else 0.0

    # energy: 1 - |song - target|  (both in [0, 1])
    scores[1] = max(0.0, 1.0 - abs(song["energy"] - prefs.get("target_energy", 0.5)))

    # valence: 1 - |song - target|
    scores[2] = max(0.0, 1.0 - abs(song["valence"] - prefs.get("target_valence", 0.5)))

    # danceability: 1 - |song - target|
    scores[3] = max(0.0, 1.0 - abs(song["danceability"] - prefs.get("target_danceability", 0.5)))

    # acousticness: binary zone check (>= 0.5 = acoustic)
    likes_acoustic = prefs.get("likes_acoustic", None)
    if likes_acoustic is None:
        scores[4] = 0.5
    else:
        scores[4] = 1.0 if (song["acousticness"] >= 0.5) == likes_acoustic else 0.0

    # tempo: normalize to [0, 1] then compute proximity
    tempo_range = max(tempo_max - tempo_min, 1.0)
    norm_tempo = (song["tempo_bpm"] - tempo_min) / tempo_range
    scores[5] = max(0.0, 1.0 - abs(norm_tempo - prefs.get("target_tempo", 0.5)))

    return scores


# ---------------------------------------------------------------------------
# Scoring and recommendation
# ---------------------------------------------------------------------------

def score_song(
    song: pd.Series,
    profile: UserProfile,
    tempo_min: float,
    tempo_max: float,
) -> float:
    """Weighted dot-product of feature scores and the user's learned weights."""
    feat = compute_feature_scores(song, profile.preferences, tempo_min, tempo_max)
    return float(np.dot(profile.weights, feat))


def recommend(
    catalog: pd.DataFrame,
    profile: UserProfile,
    k: int = 5,
) -> List[Tuple[pd.Series, float, np.ndarray]]:
    """
    Score every song and return the top-k as (song_row, score, feature_scores).

    Results are sorted by score descending.
    """
    tempo_min, tempo_max = get_tempo_bounds()
    results = []
    for _, row in catalog.iterrows():
        feat = compute_feature_scores(row, profile.preferences, tempo_min, tempo_max)
        score = float(np.dot(profile.weights, feat))
        results.append((row, score, feat))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def explain(song: pd.Series, feature_scores: np.ndarray, profile: UserProfile) -> str:
    """Plain-language explanation of why a song was recommended."""
    labels = {
        "genre": f"genre matches ({song['genre']})",
        "energy": "energy level fits",
        "valence": "mood/positivity fits",
        "danceability": "danceability fits",
        "acousticness": "acoustic feel fits",
        "tempo": "tempo fits",
    }
    reasons = [labels[name] for i, name in enumerate(FEATURE_NAMES) if feature_scores[i] >= 0.75]
    return "Because: " + ", ".join(reasons) if reasons else "Decent overall match"
