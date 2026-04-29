import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.user_profile import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    UserProfile,
    load_profile,
    record_feedback,
    save_profile,
    update_weights,
)


# ---------------------------------------------------------------------------
# UserProfile serialisation
# ---------------------------------------------------------------------------

def test_to_dict_and_from_dict_roundtrip():
    p = UserProfile(username="alice")
    p.preferences = {"favorite_genres": ["pop"], "target_energy": 0.7}
    p.weights = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])

    restored = UserProfile.from_dict(p.to_dict())
    assert restored.username == "alice"
    assert restored.preferences == p.preferences
    np.testing.assert_allclose(restored.weights, p.weights)


def test_from_dict_uses_defaults_for_missing_keys():
    p = UserProfile.from_dict({"username": "bob"})
    np.testing.assert_allclose(p.weights, DEFAULT_WEIGHTS)
    assert p.preferences == {}
    assert p.feedback_history == []
    assert p.conversation_history == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_load_profile(tmp_path, monkeypatch):
    monkeypatch.setattr("src.user_profile._USERS_DIR", tmp_path)

    profile = UserProfile(username="carol")
    profile.preferences = {"favorite_genres": ["jazz"]}
    save_profile(profile)

    loaded = load_profile("carol")
    assert loaded.username == "carol"
    assert loaded.preferences == {"favorite_genres": ["jazz"]}


def test_load_profile_creates_fresh_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.user_profile._USERS_DIR", tmp_path)
    profile = load_profile("new_user")
    assert profile.username == "new_user"
    np.testing.assert_allclose(profile.weights, DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# update_weights
# ---------------------------------------------------------------------------

def test_liked_song_increases_matching_feature_weights():
    profile = UserProfile(username="test")
    feat_scores = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    original_w0 = float(profile.weights[0])

    update_weights(profile, feat_scores, liked=True)

    assert profile.weights[0] > original_w0


def test_disliked_song_decreases_matching_feature_weights():
    profile = UserProfile(username="test")
    feat_scores = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    original_w0 = float(profile.weights[0])

    update_weights(profile, feat_scores, liked=False)

    assert profile.weights[0] < original_w0


def test_weights_always_sum_to_one_after_update():
    profile = UserProfile(username="test")
    for liked in [True, False, True, True, False]:
        feat_scores = np.random.rand(6)
        update_weights(profile, feat_scores, liked=liked)
        assert profile.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_weights_never_go_below_minimum():
    profile = UserProfile(username="test")
    # Repeatedly dislike high genre matches to try to push genre weight to zero
    feat_scores = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(100):
        update_weights(profile, feat_scores, liked=False)
    assert all(w >= 0.01 for w in profile.weights)


# ---------------------------------------------------------------------------
# record_feedback
# ---------------------------------------------------------------------------

def test_record_feedback_appends_to_history():
    profile = UserProfile(username="test")
    feat_scores = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    record_feedback(profile, song_id=1, song_title="Test Song", feature_scores=feat_scores, liked=True)

    assert len(profile.feedback_history) == 1
    entry = profile.feedback_history[0]
    assert entry["song_id"] == 1
    assert entry["song_title"] == "Test Song"
    assert entry["liked"] is True
    assert "timestamp" in entry
    assert "weights_after" in entry


def test_record_feedback_multiple_entries():
    profile = UserProfile(username="test")
    feat = np.ones(6) / 6
    record_feedback(profile, 1, "Song A", feat, liked=True)
    record_feedback(profile, 2, "Song B", feat, liked=False)
    record_feedback(profile, 3, "Song C", feat, liked=True)

    assert len(profile.feedback_history) == 3
    assert profile.feedback_history[0]["song_id"] == 1
    assert profile.feedback_history[2]["song_id"] == 3
