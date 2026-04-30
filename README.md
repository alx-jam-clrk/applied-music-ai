# Applied Music AI

An adaptive, AI-powered music recommender that improves upon the original Music Recommender Simulation by replacing its static, hand-tuned weights with a conversational preference elicitation system and a per-user online learning loop.

---

## Motivation — What Was Wrong With Fixed Weights

The original Music Recommender Simulation scored songs using a hard-coded formula:

```
score = 0.40 × genre_score
      + 0.25 × mood_score
      + 0.25 × energy_score
      + 0.10 × acoustic_score
```

These weights were chosen by intuition and never changed. That meant:

- Every user was scored the same way, regardless of what they actually cared about
- Genre always dominated — a genre match could outrank a song that was otherwise a perfect fit
- There was no way for the system to learn that a particular user cares more about tempo than danceability
- Feedback from the user (liking or skipping a song) had no effect on future recommendations

**Applied Music AI** fixes all of this. Weights start at a sensible default and shift after every thumbs-up or thumbs-down, personalizing to each user over time. Preferences are gathered through a natural conversation with Claude instead of a rigid profile form. The catalog scales from 20 songs to 81,000+.

---

## How It Works

### 1. Preference Elicitation via Claude

When a user opens the app, they chat with a Claude AI assistant that asks about their mood, genre preferences, energy level, and tempo in plain English. After 3–4 exchanges, Claude emits a structured JSON block:

```json
{
  "favorite_genres": ["chill", "indie-pop"],
  "target_energy": 0.35,
  "target_valence": 0.55,
  "target_danceability": 0.45,
  "likes_acoustic": true,
  "target_tempo": 0.4
}
```

### 2. Content Guardrails

Two guardrails are enforced at the prompt level, with a programmatic fallback as a second layer:

- **Off-topic blocking** — Claude only discusses music: songs, artists, albums, genres, moods, and recommendations. Any off-topic question (sports, cooking, news, etc.) is politely declined and redirected to music discovery.
- **Genre validation** — genres must come from the 119 entries in `data/catalog_meta.json`. If the user requests an unsupported genre (e.g. "lofi", "synthwave", "phonk"), Claude explains it isn't in the catalog and suggests the closest available options. As a programmatic safety net, `validate_genres()` in `claude_client.py` strips any unsupported genres that slip through before preferences are saved, and the UI surfaces a warning if anything was removed.

### 3. Content-Based Scoring

Each song in the catalog is scored against those preferences using six features:

- **Genre**: Binary — 1.0 if song genre is in user's list, else 0.0
- **Energy**: Proximity score — how close the song's energy is to the user's target
- **Valence**: Proximity score — how close the song's positivity/mood is to the user's target
- **Danceability**: Proximity score — how close the song's danceability is to the user's target
- **Acousticness**: Binary zone check — 1.0 if the song's acoustic character matches the user's preference
- **Tempo**: BPM normalized to [0, 1], then scored by proximity to the user's target

The final score is a weighted dot product of these six scores and the user's current weight vector.

### 4. Online Weight Learning

After every thumbs-up or thumbs-down, the weights update using a perceptron rule:

```
liked song:    weights += 0.05 × feature_scores
disliked song: weights -= 0.05 × feature_scores

→ clip to minimum 0.01
→ normalize so weights sum to 1.0
```

The starting weights are `[0.35, 0.20, 0.20, 0.10, 0.10, 0.05]` for genre, energy, valence, danceability, acousticness, and tempo. They shift with every piece of feedback and persist to disk between sessions.

---

## Project Structure

```
applied-music-ai/
├── src/
│   ├── app.py              # Streamlit web UI (main entry point)
│   ├── main.py             # CLI entry point for quick testing
│   ├── recommender.py      # Scoring engine
│   ├── user_profile.py     # Profile persistence + weight learning
│   ├── catalog.py          # Catalog loading and normalization
│   └── claude_client.py    # Anthropic API wrapper + guardrails
├── data/
│   ├── songs.csv           # 20-song fallback catalog
│   ├── catalog.parquet     # 81k-song Kaggle catalog (optional)
│   ├── catalog_meta.json   # Genre list and tempo bounds
│   └── users/              # Per-user profile JSON files
├── tests/
│   ├── test_recommender.py
│   └── test_user_profile.py
├── scripts/
│   └── preprocess_catalog.py  # ETL: Kaggle CSV → parquet
├── docs/
│   └── architecture.md     # UML diagrams
├── .env.example
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- An Anthropic API key ([get one here](https://console.anthropic.com/))

### Steps

**1. Clone the repository**

```bash
git clone <repo-url>
cd applied-music-ai
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set your API key**

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...
```

**5. (Optional) Load the large catalog**

The app ships with a 20-song fallback catalog. For the full 81,000-song dataset:

1. Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle as `tracks.csv`
2. Run the preprocessing script:

```bash
python scripts/preprocess_catalog.py path/to/tracks.csv
```

This writes `data/catalog.parquet` and `data/catalog_meta.json`. The app will automatically prefer the larger catalog when present.

---

## Running the App

### Web UI (recommended)

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`. The interface has three tabs:

- **Chat** — converse with Claude to set your preferences
- **Picks** — view top-8 recommendations, give thumbs-up/down
- **Profile** — see extracted preferences, learned weight bar chart, and full feedback history

### CLI (quick testing)

```bash
python -m src.main
```

Runs four hardcoded test profiles against the 20-song catalog and prints ranked recommendations with scores and explanations.

---

## Running Tests

```bash
pytest
```

The test suite covers:

- `tests/test_recommender.py` — feature scoring, weighted scoring, top-k ranking, explanations
- `tests/test_user_profile.py` — profile serialization, disk persistence, weight update logic

To run a specific file:

```bash
pytest tests/test_recommender.py -v
```

---

## Sample Input / Output

### Chat — Preference Elicitation

**User starts the conversation:**

> "I want something to study to"

**Claude responds:**

> "Nice! Study sessions call for the right vibe. Do you prefer something mellow and acoustic — like soft piano or folk guitar — or a more produced electronic sound like ambient synths or chill beats? And how high should the energy be — pretty quiet and relaxed, or something with a bit more drive to keep you focused?"

**User:**

> "Chill beats, pretty relaxed. Maybe 4/10 energy."

**Claude:**

> "Chill it is! Last thing — do you want something more upbeat and positive, a bit bittersweet in the middle, or leaning melancholy? And slow, medium, or faster tempo?"

**User:**

> "Bittersweet, medium tempo."

**Claude emits:**

```
Great, pulling up your recommendations now!

<preferences>
{
  "favorite_genres": ["chill"],
  "target_energy": 0.40,
  "target_valence": 0.5,
  "target_danceability": 0.4,
  "likes_acoustic": false,
  "target_tempo": 0.5
}
</preferences>
```

---

### Picks Tab — Recommendations

Initial weights `[0.35, 0.20, 0.20, 0.10, 0.10, 0.05]`:

| # | Song | Artist | Score | Why |
|---|---|---|---|---|
| 1 | Rainfall Study | Lo-Fi Collective | 0.87 | genre matches (chill), energy level fits, mood/positivity fits |
| 2 | Midnight Drift | ChillBeats | 0.81 | genre matches (chill), energy level fits |
| 3 | Soft Focus | Ambient Lab | 0.74 | energy level fits, mood/positivity fits, tempo fits |
| 4 | Late Night Pages | StudyWave | 0.71 | genre matches (chill) |
| 5 | Hazy Afternoon | The Mellow | 0.68 | energy level fits, tempo fits |

---

### Feedback — Weight Learning in Action

**User gives thumbs-down to "Late Night Pages"** (genre matched but energy didn't):

```
Before: weights = [0.35, 0.20, 0.20, 0.10, 0.10, 0.05]
Feature scores:   [1.0,  0.2,  0.5,  0.4,  0.0,  0.6 ]

After dislike:
  weights -= 0.05 × [1.0, 0.2, 0.5, 0.4, 0.0, 0.6]
  clip + normalize
  weights = [0.30, 0.21, 0.21, 0.10, 0.11, 0.05]
```

Genre weight drops; energy, valence, and acousticness nudge up. Future recommendations deprioritize pure genre matches in favor of feel.

**User gives thumbs-up to "Soft Focus"** (no genre match, but energy/valence were perfect):

```
Before: weights = [0.30, 0.21, 0.21, 0.10, 0.11, 0.05]
Feature scores:   [0.0,  0.95, 0.88, 0.55, 1.0,  0.82]

After like:
  weights += 0.05 × [0.0, 0.95, 0.88, 0.55, 1.0, 0.82]
  normalize
  weights = [0.27, 0.25, 0.24, 0.11, 0.16, 0.07]
```

Energy and acousticness now carry more weight. The system has learned this user cares more about feel than genre label.

---

### Profile Tab — Learned Weights Over Time

After 10 feedback events, the Profile tab shows a bar chart of the current weights alongside the full feedback history:

```
genre        ████░░░░░░  0.27
energy       █████░░░░░  0.25
valence      █████░░░░░  0.24
danceability ██░░░░░░░░  0.11
acousticness ███░░░░░░░  0.16
tempo        █░░░░░░░░░  0.07
```

Each feedback entry records the timestamp, song title, liked/disliked, feature scores at the time, and the resulting weights.

---

## Key Improvements Over the Original Simulation

| | Music Recommender Simulation | Applied Music AI |
|---|---|---|
| **Weights** | Fixed, hand-tuned by the developer | Learned per-user from thumbs-up/down |
| **Preferences** | Hardcoded `UserProfile` object | Extracted from natural conversation with Claude |
| **Features** | 4 (genre, mood, energy, acousticness) | 6 (genre, energy, valence, danceability, acousticness, tempo) |
| **Catalog** | 20 songs | 20-song fallback or 81k-song Kaggle dataset |
| **Personalization** | None — every user scored identically | Per-user weight vector persisted across sessions |
| **Interface** | Terminal output | Streamlit web app with chat, picks, and profile views |
| **Mood handling** | Fixed mood-valence map (hardcoded) | Continuous valence target extracted by Claude |
| **Guardrails** | None | Off-topic blocking + genre whitelist enforced at prompt and code level |

---

## Limitations

- Genre scoring is still binary — a song is either in the user's list or it is not; there is no partial credit for adjacent genres
- Supported genres are limited to the 119 entries in `catalog_meta.json`; common informal terms like "lofi", "synthwave", or "phonk" are not in the catalog and will be rejected
- Acousticness uses a fixed threshold of 0.5 to separate acoustic from electronic, which is arbitrary
- The perceptron weight update is sensitive to early feedback — a few strong dislikes can shift weights significantly before the system has much signal
- With the 20-song fallback catalog, genre coverage is narrow and may not align with the 119-genre whitelist
- Preferences are assumed stable within a session; there is no support for context-dependent recommendations (e.g., "workout mode" vs. "sleep mode")

---

## Model Card

See [model_card.md](model_card.md) for a full evaluation of the system's strengths, limitations, and potential sources of bias.
