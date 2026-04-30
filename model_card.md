# Model Card: Applied Music AI

## 1. Model Name

**SpotiVibe v2** — an adaptive, AI-assisted music recommender built on top of the original SpotiVibe v1 simulation.

---

## 2. Intended Use

SpotiVibe v2 is a content-based music recommender that suggests songs from a catalog based on preferences gathered through a natural conversation with Claude. Unlike v1, it learns from feedback: every thumbs-up or thumbs-down shifts a per-user weight vector, so recommendations improve the more a user interacts with the system.

The system is intended for personal use and educational exploration. It demonstrates how conversational AI can replace rigid profile forms, and how online learning can replace fixed weights. It is not a production-grade recommender — the catalog is limited and the learning algorithm is simple — but it mirrors the structure of real adaptive recommenders like Spotify's Discover Weekly in a transparent, inspectable way.

---

## 3. How the Model Works

SpotiVibe v2 scores every song in the catalog against a user profile and returns the top matches. The pipeline has three stages.

**Stage 1 — Preference Elicitation with Guardrails**

Instead of filling out a form, the user has a short conversation with a Claude AI assistant. Claude asks about genres, energy level, mood/vibe, acoustic preference, and tempo across 3–4 exchanges, then emits a structured JSON block with six numeric targets. This replaces the hardcoded `UserProfile` objects from v1 and allows preferences to be expressed naturally ("something to study to, not too intense") rather than as exact numbers.

Two guardrails operate during this stage. First, Claude is instructed to only discuss music-related topics; off-topic questions are politely declined and redirected. Second, the full list of 119 valid genres from `catalog_meta.json` is injected into the system prompt; if the user requests a genre outside that list (e.g. "lofi", "synthwave", "phonk"), Claude explains it is not in the catalog and suggests the closest supported alternatives. As a second line of defence, `claude_client.validate_genres()` strips any unsupported genres that slip through the model before preferences are persisted.

**Stage 2 — Content-Based Scoring**

Each song is evaluated across six features, each producing a score in [0, 1]:

- **Genre**: Binary — 1.0 if the song's genre is in the user's preferred list, else 0.0
- **Energy**: Proximity score — how close the song's energy level is to the user's target
- **Valence**: Proximity score — how close the song's positivity/mood is to the user's target
- **Danceability**: Proximity score — how close the song's danceability is to the user's target
- **Acousticness**: Binary zone check — 1.0 if the song's acoustic character matches the user's preference (threshold: 0.5)
- **Tempo**: BPM normalized to [0, 1], then scored by proximity to the user's target

The final score is a weighted dot product of these six feature scores and the user's current weight vector.

**Stage 3 — Online Weight Learning**

After every thumbs-up or thumbs-down, weights update using a perceptron rule:

```
liked:    weights += 0.05 × feature_scores
disliked: weights -= 0.05 × feature_scores
→ clip to minimum 0.01 per feature
→ normalize so all weights sum to 1.0
```

Weights start at `[0.35, 0.20, 0.20, 0.10, 0.10, 0.05]` for genre, energy, valence, danceability, acousticness, and tempo. They drift toward features that distinguished liked songs and away from features that dominated disliked ones. Weights persist to disk between sessions.

---

## 4. Data

The system ships with two catalogs:

**Fallback catalog** (`data/songs.csv`) — 20 songs across 7 genres and 6 moods (happy, chill, intense, relaxed, focused, moody). This is the same dataset from v1, retained for testing and offline use. Note that the genre labels in this file use informal names that may not match the 119-genre whitelist enforced during conversation; it is intended for development and offline testing only.

**Full catalog** (`data/catalog.parquet`) — 81,343 songs sourced from the [Kaggle Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset), covering 118 genres. Each song has: title, artist, genre, energy, tempo (BPM), valence, danceability, and acousticness. The app uses this catalog when the parquet file is present, falling back to the CSV otherwise.

Each song attribute (except tempo) is normalized to [0, 1]. Tempo is normalized at scoring time using the catalog's min/max bounds stored in `data/catalog_meta.json`.

The 20-song fallback still reflects the same biases as v1: no hip-hop, R&B, classical, or country; genre/mood combinations are unevenly distributed. The full Kaggle catalog is far broader but inherits whatever biases exist in Spotify's popularity-weighted data collection.

---

## 5. Strengths

**Personalization that actually works.** The biggest weakness of v1 was that every user was scored identically regardless of what they actually cared about. In v2, a user who consistently likes songs regardless of genre but always thumbs-down anything with high danceability will see those preferences reflected in their weights within a handful of feedback events — without ever having to say "I care less about danceability."

**Natural preference entry.** Asking a user to set `target_energy: 0.65` is not how people think about music. Claude translates plain-language descriptions ("pretty chill, maybe 6/10 energy") into numeric targets. This makes the system accessible to anyone and avoids the issue in v1 where preference parameters had to be hardcoded by the developer.

**Transparent scoring and learning.** The Profile tab shows the current weight vector as a bar chart and logs every feedback event with timestamps, feature scores, and the resulting weights. The reasoning behind each recommendation is surfaced in plain language ("genre matches, energy level fits"). Users can see exactly what the system learned from them and why a song was recommended.

**Scalable catalog.** With the full Kaggle dataset, the system can serve preferences that the 20-song catalog couldn't handle at all — a hip-hop fan no longer gets a genre score of 0 on every song.

---

## 6. Limitations and Bias

**Genre is still binary.** A song is either in the user's genre list or it is not. There is no partial credit for adjacent genres (e.g., "indie pop" and "pop" score completely independently). Users who like a broad mix of related genres are not well-served unless they enumerate every genre explicitly during the conversation.

**Acousticness threshold is arbitrary.** The 0.5 cutoff for acoustic vs. electronic is a carry-over from v1. Songs near the boundary are sharply penalized even if they are close to the user's preference, while songs far from the threshold on the "right" side get the same score as those perfectly on the edge.

**Early feedback has an outsized effect.** The perceptron update applies the same learning rate (0.05) regardless of how much data the system has seen. The first few thumbs-down events can shift weights significantly before the system has enough signal to be reliable. This is most noticeable in short sessions.

**No mood feature.** v1 had an explicit mood label match (e.g. "chill", "intense"). v2 removed mood as a direct feature and replaced it with continuous valence scoring extracted by Claude. This is more flexible, but it means the system can no longer distinguish between two songs with identical valence values that evoke very different moods. A "focused" instrumental and a "melancholy" ballad might score identically against the same valence target.

**Preferences are session-constant.** The system assumes that once preferences are elicited, they stay fixed throughout the session. There is no support for context-dependent listening (e.g., "gym mode" vs. "winding down"). A user whose energy preference shifts mid-session has no way to signal that other than restarting the conversation.

**Genre whitelist is narrow.** The 119 genres in the catalog cover Spotify's formal taxonomy but omit many informal or emerging terms that users naturally reach for — "lofi", "synthwave", "phonk", "bedroom pop", and many others will be rejected. Users unfamiliar with Spotify's genre vocabulary may find the system frustrating until they learn what terms are accepted.

**Kaggle catalog biases.** The full catalog is sourced from Spotify streaming data, which over-represents mainstream genres and popular artists. Niche or regional music is underrepresented, so recommendations will systematically skew toward commercially popular styles even when other features match perfectly. The dataset also has no filter for language or region — a song's genre tag (e.g. "anime") reflects the style label assigned in Spotify's metadata, not the language the song is sung in or the culture it comes from. A user asking for "anime" music could receive songs entirely in Spanish, or a user requesting "pop" could receive tracks from any country. The system has no way to distinguish or respect those preferences.

---

## 7. Evaluation

The same four adversarial profiles from v1 were re-run against v2 to benchmark the improvement.

**The Impossible Profile** (`chill, pop / relaxed / energy 0.35 / acoustic`): In v1, this profile was hardcoded and the weights were fixed. In v2, Claude correctly extracted the preferences from a natural description ("something soft and chill to study to") and mapped them to the supported `chill` genre. The recommendations surfaced matches immediately. Unlike v1, when the user thumbed-down a genre match that felt too energetic, subsequent picks correctly deprioritized energy-heavy songs — the system learned rather than repeating the same mistake.

**The Energy Paradox** (`chill / relaxed / energy 0.0 / acoustic`): v1 failed this profile because the composite energy formula couldn't reach 0.0, always disadvantaging true low-energy listeners. v2 separates energy, danceability, and tempo as independent features with independent weights, removing the forced composite. A user who dislikes high-danceability songs can now push that weight down without affecting the energy score.

**The Genre Intruder** (`synth-pop / chill / energy 0.40 / acoustic`): This was the most important test in v1 — a single genre match dominated over every other signal. In v2 the starting genre weight is 0.35 (down from 0.40 in the rebalanced v1), and crucially, it shifts. After a few thumbs-downs on pure genre matches that missed on feel, the genre weight drops below 0.30 while energy and acousticness rise. The system self-corrects what previously required manual weight tuning.

**The Valence Trap** (`jazz / happy / energy 0.40 / acoustic`): In v1 this exposed how valence similarity could silently inflate scores for songs with no mood label match. In v2, valence is an explicit top-level feature with its own weight, not buried inside a mood composite. The behavior is more interpretable — if valence is scoring a song high, the explanation says "mood/positivity fits" directly.

**New test — Feedback Convergence**: After 10 feedback events on a mixed session (5 likes, 5 dislikes), the learned weights were compared against the initial defaults. In all test runs, features that were consistently high-scoring on liked songs and low-scoring on disliked songs gained weight, while the reverse pattern caused weight reduction. The system converged toward a user-specific weight vector within 10–15 interactions.

The most significant finding from evaluation was that the online learning loop made the Genre Intruder problem self-correcting. In v1 it required manual tuning; in v2 it resolves on its own after a few feedback events.

---

## 8. Future Work

**Smarter learning algorithm.** The current perceptron update applies the same learning rate to every feedback event regardless of how confident the system is. A Bayesian or bandit-style approach would down-weight early feedback and up-weight signals from later sessions when more data is available. This would reduce the instability in short sessions.

**Multi-genre partial credit.** The binary genre score is the single largest source of poor recommendations. A similarity graph over genres (e.g., "indie pop" is adjacent to "pop" and "dream pop") would allow partial credit and serve listeners with broad or overlapping tastes.

**Language and region filtering.** The Kaggle catalog assigns genre labels based on Spotify's metadata, which can be inconsistent — a song tagged "anime" may be entirely in Spanish, and a user requesting "j-pop" may receive tracks from multiple countries. Adding language detection or a region filter would make the system significantly more useful for listeners who care about the language a song is sung in.

**Context-aware sessions.** Supporting multiple "modes" per user (workout, focus, sleep, social) would let the system maintain separate weight vectors for different contexts and switch between them based on a simple signal at session start.

**Collaborative signals.** The current system is purely content-based — it has no knowledge of what other users like. Adding a lightweight collaborative layer (e.g., "users with similar weight vectors also liked...") could surface songs that score low on features but are empirically well-liked by similar listeners.

**Evaluation metric.** Right now there is no numeric measure of recommendation quality. Adding a held-out test set and measuring precision@k or NDCG over a set of labeled preferences would make it possible to compare algorithm changes objectively rather than relying on manual profile testing.

---

## 9. Personal Reflection

Building v2 answered the question I left open at the end of v1: what would it actually take to make the weights non-fixed? It turned out to be less code than I expected — the perceptron update is about 5 lines — but the design decisions around *what* to update, *when*, and *how fast* took much more thought. The learning rate, the minimum weight floor, and the normalization step are all choices that meaningfully change how quickly the system responds to feedback versus how stable it is over time.

The other big lesson was how much the preference elicitation step matters. In v1, a "bad" profile (like Genre Intruder) was something I constructed manually to expose a weakness. In v2, Claude sometimes extracts a preference that doesn't quite capture what the user meant — for instance, a user saying "upbeat but not intense" might get a high valence target but a low energy target, which are not the same thing. The quality of the conversation directly determines the quality of the starting point, and no amount of learning can fully recover from a badly initialized preference vector. That's a problem Spotify and every other real recommender faces too: cold-start is genuinely hard, and the interface you use to gather initial preferences is a core part of the system, not just a wrapper around it.

The most surprising thing overall was how quickly the weight learning made Genre Intruder-style problems disappear on their own. In v1 I had to manually experiment to fix that bias. In v2 a few thumbs-downs on genre matches that missed on feel naturally pushed the genre weight down. Watching the bar chart shift in the Profile tab made the learning loop feel real in a way that I didn't expect from such a simple algorithm.

Working with the Kaggle catalog also surfaced a limitation I hadn't considered before: genre labels in real datasets are messy and culturally inconsistent. A song tagged "anime" might be in Spanish; "pop" might mean K-pop, Latin pop, or American Top 40 depending on the metadata source. The system has no way to handle that — it treats genre as a clean label when the underlying data is far noisier than that. That's a gap I'd want to close in any real deployment.
