import numpy as np
import pandas as pd
import pytest

from src.recommender import compute_feature_scores, explain, recommend, score_song
from src.user_profile import DEFAULT_WEIGHTS, UserProfile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEMPO_MIN = 60.0
TEMPO_MAX = 200.0


def make_song(**kwargs) -> pd.Series:
    defaults = {
        "id": 1,
        "title": "Test Track",
        "artist": "Test Artist",
        "genre": "pop",
        "energy": 0.8,
        "tempo_bpm": 120.0,
        "valence": 0.8,
        "danceability": 0.75,
        "acousticness": 0.2,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def make_profile(**prefs) -> UserProfile:
    defaults = {
        "favorite_genres": ["pop"],
        "target_energy": 0.8,
        "target_valence": 0.8,
        "target_danceability": 0.75,
        "likes_acoustic": False,
        "target_tempo": 0.46,  # ~(120 - 60) / 130 ≈ 0.46
    }
    defaults.update(prefs)
    p = UserProfile(username="test")
    p.preferences = defaults
    return p


def make_catalog() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": 1, "title": "Pop Hit", "artist": "A",
            "genre": "pop", "energy": 0.8, "tempo_bpm": 120.0,
            "valence": 0.8, "danceability": 0.75, "acousticness": 0.2,
        },
        {
            "id": 2, "title": "Chill Lofi", "artist": "B",
            "genre": "lofi", "energy": 0.3, "tempo_bpm": 75.0,
            "valence": 0.5, "danceability": 0.5, "acousticness": 0.8,
        },
        {
            "id": 3, "title": "Rock Anthem", "artist": "C",
            "genre": "rock", "energy": 0.9, "tempo_bpm": 150.0,
            "valence": 0.4, "danceability": 0.6, "acousticness": 0.1,
        },
    ])


# ---------------------------------------------------------------------------
# compute_feature_scores
# ---------------------------------------------------------------------------

def test_genre_match_scores_one():
    song = make_song(genre="pop")
    prefs = {"favorite_genres": ["pop"], "target_energy": 0.5, "target_valence": 0.5,
             "target_danceability": 0.5, "likes_acoustic": None, "target_tempo": 0.5}
    scores = compute_feature_scores(song, prefs, TEMPO_MIN, TEMPO_MAX)
    assert scores[0] == pytest.approx(1.0)


def test_genre_mismatch_scores_zero():
    song = make_song(genre="rock")
    prefs = {"favorite_genres": ["pop"], "target_energy": 0.5, "target_valence": 0.5,
             "target_danceability": 0.5, "likes_acoustic": None, "target_tempo": 0.5}
    scores = compute_feature_scores(song, prefs, TEMPO_MIN, TEMPO_MAX)
    assert scores[0] == pytest.approx(0.0)


def test_energy_proximity_score():
    song = make_song(energy=0.8)
    prefs = {"favorite_genres": [], "target_energy": 0.8, "target_valence": 0.5,
             "target_danceability": 0.5, "likes_acoustic": None, "target_tempo": 0.5}
    scores = compute_feature_scores(song, prefs, TEMPO_MIN, TEMPO_MAX)
    assert scores[1] == pytest.approx(1.0)


def test_acousticness_binary_check():
    acoustic_song = make_song(acousticness=0.8)
    electric_song = make_song(acousticness=0.2)
    prefs = {"favorite_genres": [], "target_energy": 0.5, "target_valence": 0.5,
             "target_danceability": 0.5, "likes_acoustic": True, "target_tempo": 0.5}

    scores_acoustic = compute_feature_scores(acoustic_song, prefs, TEMPO_MIN, TEMPO_MAX)
    scores_electric = compute_feature_scores(electric_song, prefs, TEMPO_MIN, TEMPO_MAX)
    assert scores_acoustic[4] == pytest.approx(1.0)
    assert scores_electric[4] == pytest.approx(0.0)


def test_all_scores_in_unit_interval():
    song = make_song()
    prefs = {"favorite_genres": ["pop"], "target_energy": 0.8, "target_valence": 0.8,
             "target_danceability": 0.75, "likes_acoustic": False, "target_tempo": 0.5}
    scores = compute_feature_scores(song, prefs, TEMPO_MIN, TEMPO_MAX)
    assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# score_song
# ---------------------------------------------------------------------------

def test_perfect_match_scores_high():
    song = make_song(genre="pop", energy=0.8, valence=0.8, danceability=0.75, acousticness=0.2, tempo_bpm=120.0)
    profile = make_profile()
    s = score_song(song, profile, TEMPO_MIN, TEMPO_MAX)
    assert s > 0.7


def test_genre_mismatch_reduces_score():
    matching = make_song(genre="pop")
    wrong_genre = make_song(genre="rock")
    profile = make_profile()
    assert score_song(matching, profile, TEMPO_MIN, TEMPO_MAX) > score_song(wrong_genre, profile, TEMPO_MIN, TEMPO_MAX)


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

def test_recommend_returns_k_results(monkeypatch):
    monkeypatch.setattr("src.recommender.get_tempo_bounds", lambda: (TEMPO_MIN, TEMPO_MAX))
    catalog = make_catalog()
    profile = make_profile()
    results = recommend(catalog, profile, k=2)
    assert len(results) == 2


def test_recommend_sorted_descending(monkeypatch):
    monkeypatch.setattr("src.recommender.get_tempo_bounds", lambda: (TEMPO_MIN, TEMPO_MAX))
    catalog = make_catalog()
    profile = make_profile()
    results = recommend(catalog, profile, k=3)
    scores = [s for _, s, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_pop_ranks_first_for_pop_user(monkeypatch):
    monkeypatch.setattr("src.recommender.get_tempo_bounds", lambda: (TEMPO_MIN, TEMPO_MAX))
    catalog = make_catalog()
    profile = make_profile(favorite_genres=["pop"])
    results = recommend(catalog, profile, k=3)
    top_song = results[0][0]
    assert top_song["genre"] == "pop"


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

def test_explain_returns_string(monkeypatch):
    monkeypatch.setattr("src.recommender.get_tempo_bounds", lambda: (TEMPO_MIN, TEMPO_MAX))
    song = make_song()
    profile = make_profile()
    feat = compute_feature_scores(song, profile.preferences, TEMPO_MIN, TEMPO_MAX)
    result = explain(song, feat, profile)
    assert isinstance(result, str)
    assert result.strip()
