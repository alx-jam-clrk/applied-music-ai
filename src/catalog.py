"""
catalog.py — Song catalog loader.

Tries to load the preprocessed Kaggle-derived catalog.parquet first.
Falls back to the legacy data/songs.csv (20 songs) if parquet is not found.
Results are cached so the DataFrame is only read once per Streamlit session.
"""

import json
import os
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
_PARQUET_PATH = _ROOT / "data" / "catalog.parquet"
_META_PATH = _ROOT / "data" / "catalog_meta.json"
_CSV_FALLBACK = _ROOT / "data" / "songs.csv"

# Tempo normalization bounds (updated from catalog_meta.json when available)
TEMPO_MIN = 60.0
TEMPO_MAX = 220.0


def load_catalog() -> pd.DataFrame:
    """
    Load the song catalog as a DataFrame.

    Columns returned: id, title, artist, genre, energy, tempo_bpm,
    valence, danceability, acousticness.

    Uses Streamlit cache when called from the app; falls back to a
    plain lru_cache-style singleton for tests and scripts.
    """
    try:
        import streamlit as st
        return _load_cached_st()
    except Exception:
        return _load_raw()


try:
    import streamlit as st

    @st.cache_data
    def _load_cached_st() -> pd.DataFrame:
        return _load_raw()

except ImportError:
    def _load_cached_st() -> pd.DataFrame:
        return _load_raw()


def _load_raw() -> pd.DataFrame:
    global TEMPO_MIN, TEMPO_MAX

    if _PARQUET_PATH.exists():
        df = pd.read_parquet(_PARQUET_PATH)
        meta = load_catalog_meta()
        TEMPO_MIN = meta.get("tempo_min", TEMPO_MIN)
        TEMPO_MAX = meta.get("tempo_max", TEMPO_MAX)
    else:
        # Fall back to the legacy 20-song CSV
        df = pd.read_csv(_CSV_FALLBACK)
        # Drop mood column if present (not used in new scoring)
        if "mood" in df.columns:
            df = df.drop(columns=["mood"])
        TEMPO_MIN = float(df["tempo_bpm"].min())
        TEMPO_MAX = float(df["tempo_bpm"].max())

    # Normalize genre strings: lowercase and strip whitespace
    df["genre"] = df["genre"].str.lower().str.strip()

    # Ensure required columns exist and types are correct
    float_cols = ["energy", "tempo_bpm", "valence", "danceability", "acousticness"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0 if col != "tempo_bpm" else None)

    df = df.dropna(subset=float_cols)
    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index

    return df


def load_catalog_meta() -> dict:
    """Load metadata (genres list, tempo bounds) from catalog_meta.json."""
    if _META_PATH.exists():
        with open(_META_PATH, encoding="utf-8") as f:
            return json.load(f)

    # Generate meta from the CSV fallback
    df = pd.read_csv(_CSV_FALLBACK)
    genres = sorted(df["genre"].str.lower().str.strip().unique().tolist())
    return {
        "genres": genres,
        "tempo_min": float(df["tempo_bpm"].min()),
        "tempo_max": float(df["tempo_bpm"].max()),
        "n_songs": len(df),
    }


def get_genres() -> list:
    """Return a sorted list of unique genre strings from the catalog."""
    return load_catalog_meta().get("genres", [])


def get_tempo_bounds() -> tuple:
    """Return (tempo_min, tempo_max) for normalization."""
    meta = load_catalog_meta()
    return meta.get("tempo_min", TEMPO_MIN), meta.get("tempo_max", TEMPO_MAX)
