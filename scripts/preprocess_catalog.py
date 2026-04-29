"""
preprocess_catalog.py — Convert a Kaggle Spotify tracks CSV to catalog.parquet.

Usage:
    python scripts/preprocess_catalog.py path/to/tracks.csv
    python scripts/preprocess_catalog.py path/to/tracks.csv --sample 5000

The recommended dataset is "Spotify Tracks Dataset" from Kaggle (114k songs):
    https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Expected CSV columns:
    track_name, artists, track_genre, energy, tempo,
    valence, danceability, acousticness

Outputs:
    data/catalog.parquet
    data/catalog_meta.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
_OUT_PARQUET = _ROOT / "data" / "catalog.parquet"
_OUT_META = _ROOT / "data" / "catalog_meta.json"

COLUMN_MAP = {
    "track_name": "title",
    "artists": "artist",
    "track_genre": "genre",
    "energy": "energy",
    "tempo": "tempo_bpm",
    "valence": "valence",
    "danceability": "danceability",
    "acousticness": "acousticness",
}


def preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()
    df["genre"] = df["genre"].str.lower().str.strip()

    float_cols = ["energy", "valence", "danceability", "acousticness"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)

    df["tempo_bpm"] = pd.to_numeric(df["tempo_bpm"], errors="coerce")

    df = df.dropna().drop_duplicates(subset=["title", "artist"]).reset_index(drop=True)
    df["id"] = df.index

    return df


def build_meta(df: pd.DataFrame) -> dict:
    return {
        "genres": sorted(df["genre"].unique().tolist()),
        "tempo_min": float(df["tempo_bpm"].min()),
        "tempo_max": float(df["tempo_bpm"].max()),
        "n_songs": len(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Kaggle Spotify CSV → parquet")
    parser.add_argument("csv_path", help="Path to the Kaggle tracks CSV file")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N songs (optional, useful for testing)"
    )
    args = parser.parse_args()

    print(f"Reading {args.csv_path} ...")
    df = preprocess(args.csv_path)

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        df["id"] = df.index

    print(f"Processed {len(df):,} songs across {df['genre'].nunique()} genres.")

    _OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT_PARQUET, index=False)
    print(f"Saved → {_OUT_PARQUET}")

    meta = build_meta(df)
    with open(_OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved → {_OUT_META}")

    sample_genres = meta["genres"][:8]
    suffix = "..." if len(meta["genres"]) > 8 else ""
    print(f"Genres: {sample_genres}{suffix}")


if __name__ == "__main__":
    main()
