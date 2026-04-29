"""
app.py — Streamlit music recommender app.

Three tabs:
  1. Chat    — conversational preference elicitation with Claude
  2. Picks   — top-k recommendations with thumbs-up / thumbs-down feedback
  3. Profile — learned weight bar chart and feedback history
"""

import re

import numpy as np
import pandas as pd
import streamlit as st

from src.catalog import load_catalog
from src.claude_client import chat, extract_preferences
from src.recommender import explain, recommend
from src.user_profile import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    load_profile,
    record_feedback,
    save_profile,
)

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = load_profile("default")
    if "prefs_ready" not in st.session_state:
        st.session_state.prefs_ready = bool(st.session_state.profile.preferences)
    if "recs" not in st.session_state:
        st.session_state.recs = []  # list of (song_dict, score, feat_list)


_init_state()


def _profile():
    return st.session_state.profile


# ---------------------------------------------------------------------------
# Sidebar — username switcher
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎵 Music Recommender")
    username = st.text_input("Username", value=_profile().username)

    if username != _profile().username:
        st.session_state.profile = load_profile(username)
        st.session_state.prefs_ready = bool(st.session_state.profile.preferences)
        st.session_state.recs = []
        st.rerun()

    st.caption(f"Feedback given: {len(_profile().feedback_history)}")

    if st.button("Reset this profile"):
        from src.user_profile import UserProfile
        st.session_state.profile = UserProfile(username=_profile().username)
        st.session_state.prefs_ready = False
        st.session_state.recs = []
        save_profile(st.session_state.profile)
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_picks, tab_profile = st.tabs(["💬 Chat", "🎶 Picks", "📊 Profile"])


# ---- TAB 1: CHAT -----------------------------------------------------------

with tab_chat:
    st.header("Tell me what you're in the mood for")
    profile = _profile()

    # Render existing conversation (strip the hidden <preferences> block)
    for msg in profile.conversation_history:
        with st.chat_message(msg["role"]):
            visible = re.sub(r"<preferences>.*?</preferences>", "", msg["content"], flags=re.DOTALL).strip()
            st.write(visible)

    # Kick off with an opening line if the history is empty
    if not profile.conversation_history:
        with st.spinner("Starting conversation..."):
            opening = chat([{"role": "user", "content": "Hi, I want music recommendations."}])
        profile.conversation_history.append({"role": "user", "content": "Hi, I want music recommendations."})
        profile.conversation_history.append({"role": "assistant", "content": opening})
        save_profile(profile)
        st.rerun()

    if st.session_state.prefs_ready:
        st.success("Preferences captured — head to the **Picks** tab!")
    else:
        user_input = st.chat_input("Your answer...")
        if user_input:
            profile.conversation_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                reply = chat(profile.conversation_history)
            profile.conversation_history.append({"role": "assistant", "content": reply})

            prefs = extract_preferences(reply)
            if prefs:
                profile.preferences = prefs
                st.session_state.prefs_ready = True

            save_profile(profile)
            st.rerun()


# ---- TAB 2: PICKS ----------------------------------------------------------

with tab_picks:
    st.header("Your Picks")
    profile = _profile()

    if not st.session_state.prefs_ready:
        st.info("Complete the **Chat** tab first to get personalised recommendations.")
    else:
        catalog = load_catalog()

        if st.button("🔄 Refresh") or not st.session_state.recs:
            raw = recommend(catalog, profile, k=8)
            # Store as JSON-serialisable tuples so Streamlit doesn't choke on numpy
            st.session_state.recs = [
                (row.to_dict(), float(score), feat.tolist())
                for row, score, feat in raw
            ]

        for i, (song_dict, score, feat_list) in enumerate(st.session_state.recs):
            song = pd.Series(song_dict)
            feat = np.array(feat_list)

            col_info, col_btns = st.columns([5, 1])
            with col_info:
                st.markdown(f"**{song['title']}** — {song.get('artist', 'Unknown')}")
                st.caption(f"Genre: {song['genre']} | Score: {score:.2f}")
                st.caption(explain(song, feat, profile))
            with col_btns:
                if st.button("👍", key=f"like_{i}_{song['id']}"):
                    record_feedback(profile, int(song["id"]), song["title"], feat, liked=True)
                    save_profile(profile)
                    raw = recommend(catalog, profile, k=8)
                    st.session_state.recs = [
                        (r.to_dict(), float(s), f.tolist()) for r, s, f in raw
                    ]
                    st.rerun()
                if st.button("👎", key=f"dislike_{i}_{song['id']}"):
                    record_feedback(profile, int(song["id"]), song["title"], feat, liked=False)
                    save_profile(profile)
                    raw = recommend(catalog, profile, k=8)
                    st.session_state.recs = [
                        (r.to_dict(), float(s), f.tolist()) for r, s, f in raw
                    ]
                    st.rerun()

            st.divider()


# ---- TAB 3: PROFILE --------------------------------------------------------

with tab_profile:
    st.header("Your Taste Profile")
    profile = _profile()

    if profile.preferences:
        st.subheader("Extracted Preferences")
        st.json(profile.preferences)
    else:
        st.info("No preferences yet — complete the Chat tab first.")

    st.subheader("Learned Feature Weights")
    weight_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Weight": profile.weights})
    st.bar_chart(weight_df.set_index("Feature"))

    if st.button("Reset weights to default"):
        profile.weights = DEFAULT_WEIGHTS.copy()
        save_profile(profile)
        st.rerun()

    if profile.feedback_history:
        st.subheader("Recent Feedback")
        rows = [
            {
                "Song": h["song_title"],
                "Reaction": "👍" if h["liked"] else "👎",
                "Time": h["timestamp"][:19].replace("T", " "),
            }
            for h in reversed(profile.feedback_history[-20:])
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("Rate songs in the Picks tab to see your history here.")
