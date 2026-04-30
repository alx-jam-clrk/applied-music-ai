# System Architecture — Applied Music AI

## Component Diagram

```mermaid
graph TD
    subgraph EntryPoints["Entry Points"]
        APP["app.py\nStreamlit Web UI"]
        MAIN["main.py\nCLI"]
    end

    subgraph Core["Core Modules"]
        CC["claude_client.py\nchat()\nextract_preferences()\nvalidate_genres()"]
        REC["recommender.py\ncompute_feature_scores()\nscore_song()\nrecommend()\nexplain()"]
        UP["user_profile.py\nload_profile()\nsave_profile()\nupdate_weights()\nrecord_feedback()"]
        CAT["catalog.py\nload_catalog()\nload_catalog_meta()\nget_genres()\nget_tempo_bounds()"]
    end

    subgraph External["External Services"]
        API["Anthropic API\nClaude Haiku"]
    end

    subgraph Data["Data Layer"]
        PQ["catalog.parquet\n81k songs"]
        CSV["songs.csv\n20 songs fallback"]
        META["catalog_meta.json\ngenres, tempo bounds"]
        PROF["users/{name}/profile.json\npreferences, weights, history"]
    end

    APP --> CC
    APP --> REC
    APP --> UP
    APP --> CAT
    MAIN --> REC
    MAIN --> CAT

    CC --> API
    CC --> META
    UP --> PROF
    CAT --> PQ
    CAT --> CSV
    CAT --> META
```

---

## Class Diagram

```mermaid
classDiagram
    class App {
        +_init_state()
        +chat_tab()
        +picks_tab()
        +profile_tab()
    }

    class ClaudeClient {
        +VALID_GENRES : list
        +chat(messages, max_tokens) str
        +extract_preferences(text) dict
        +validate_genres(prefs) tuple
    }

    class Recommender {
        +compute_feature_scores(song, prefs, tempo_min, tempo_max) ndarray
        +score_song(song, profile, tempo_min, tempo_max) float
        +recommend(catalog, profile, k) list
        +explain(song, feature_scores, profile) str
    }

    class UserProfile {
        +username : str
        +preferences : dict
        +weights : ndarray
        +feedback_history : list
        +conversation_history : list
        +to_dict() dict
        +from_dict(d) UserProfile
    }

    class UserProfileModule {
        +load_profile(username) UserProfile
        +save_profile(profile)
        +update_weights(profile, feature_scores, liked)
        +record_feedback(profile, song_id, song_title, feature_scores, liked)
    }

    class Catalog {
        +load_catalog() DataFrame
        +load_catalog_meta() dict
        +get_genres() list
        +get_tempo_bounds() tuple
    }

    class AnthropicAPI {
        <<external>>
        +messages.create()
    }

    class DataLayer {
        <<datastore>>
        catalog.parquet
        songs.csv
        catalog_meta.json
        users_profile.json
    }

    App --> ClaudeClient : elicits preferences
    App --> Recommender : scores songs
    App --> UserProfileModule : loads & saves state
    App --> Catalog : loads song catalog
    App --> UserProfile : reads/writes session state

    ClaudeClient --> AnthropicAPI : API call
    UserProfileModule --> UserProfile : creates & mutates
    UserProfileModule --> DataLayer : persists JSON
    Recommender --> UserProfile : reads weights & prefs
    Catalog --> DataLayer : reads parquet / CSV
```

---

## Sequence Diagram — Feedback Loop

```mermaid
sequenceDiagram
    actor User
    participant App as app.py
    participant Claude as claude_client.py
    participant API as Anthropic API
    participant Rec as recommender.py
    participant Cat as catalog.py
    participant UP as user_profile.py
    participant Disk as data/users/

    User->>App: Enter chat message
    App->>Claude: chat(messages)
    Claude->>API: POST /messages
    API-->>Claude: response text
    Claude-->>App: assistant reply
    App->>Claude: extract_preferences(reply)
    Claude-->>App: preferences dict (or None)
    App->>Claude: validate_genres(prefs)
    Claude-->>App: cleaned prefs + rejected genres list
    Note over App: shows warning if any genres were rejected

    User->>App: Click "Get Recommendations"
    App->>Cat: load_catalog()
    Cat-->>App: DataFrame (81k songs)
    App->>Rec: recommend(catalog, profile, k=8)
    Rec-->>App: top-k (song, score, feature_scores)
    App-->>User: Display recommendations

    User->>App: Thumbs up / down
    App->>UP: record_feedback(profile, song_id, liked)
    UP->>UP: update_weights(profile, feature_scores, liked)
    UP->>Disk: save_profile(profile)
    Disk-->>UP: OK
    App->>Rec: recommend(catalog, profile, k=8)
    Rec-->>App: updated top-k
    App-->>User: Refreshed recommendations
```

---

## Scoring Pipeline

```mermaid
flowchart LR
    SONG["Song\n(genre, energy,\nvalence, danceability,\nacousticness, tempo)"]
    PREFS["User Preferences\n(favorite_genres,\ntarget_energy,\ntarget_valence,\ntarget_danceability,\nlikes_acoustic,\ntarget_tempo)"]
    FS["Feature Scores\n[0,1] × 6"]
    W["Learned Weights\n[0.35, 0.20, 0.20,\n0.10, 0.10, 0.05]"]
    SCORE["Final Score\n(dot product)"]
    TOPK["Top-k Songs"]
    EXPLAIN["Plain-language\nExplanation"]

    SONG --> FS
    PREFS --> FS
    FS --> SCORE
    W --> SCORE
    SCORE --> TOPK
    FS --> EXPLAIN
    W --> EXPLAIN
```

---

## Weight Learning (Online Perceptron)

```mermaid
flowchart TD
    FB["User Feedback\n(liked / disliked)"]
    FS2["Feature Scores\nof rated song"]
    SIGNAL{liked?}
    ADD["weights += 0.05 × feature_scores"]
    SUB["weights -= 0.05 × feature_scores"]
    CLIP["Clip to min 0.01"]
    NORM["Normalize to sum = 1.0"]
    SAVE["Save profile to disk"]

    FB --> SIGNAL
    FS2 --> ADD
    FS2 --> SUB
    SIGNAL -- yes --> ADD
    SIGNAL -- no --> SUB
    ADD --> CLIP
    SUB --> CLIP
    CLIP --> NORM
    NORM --> SAVE
```
