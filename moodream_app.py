import streamlit as st
import pandas as pd
import datetime
import os
import json
import re
import numpy as np

from textblob import TextBlob
from nrclex import NRCLex
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


DATA_FILE = "dreams.json"


# --- Utility Functions ---

def load_data():
    """Load dreams data from JSON file into DataFrame."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['date'] = pd.NaT
    else:
        # Empty DataFrame with expected columns
        df = pd.DataFrame(columns=[
            'date', 'dream', 'mood_score', 'mood_label', 'type', 'topics', 'mood_detail'
        ])
    return df


def save_data(df):
    """Save DataFrame to JSON file with date as string."""
    df_copy = df.copy()
    df_copy['date'] = df_copy['date'].astype(str)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(df_copy.to_dict(orient='records'), f, ensure_ascii=False, indent=4)


def analyze_mood_multi(text):
    """Analyze text mood using NRCLex and return normalized scores."""
    emotion = NRCLex(text)
    scores = emotion.raw_emotion_scores
    moods = ["joy", "fear", "anger", "sadness", "surprise", "trust"]

    if not scores:
        return {m: 0.0 for m in moods}

    total = sum(scores.values())
    normalized = {m: scores.get(m, 0) / total for m in moods}
    return normalized


def get_weighted_mood_score(mood_detail):
    """Calculate weighted mood score based on predefined weights."""
    weights = {
        "joy": 1.0,
        "trust": 0.8,
        "surprise": 0.3,
        "sadness": -0.7,
        "fear": -1.0,
        "anger": -0.9,
    }
    return sum(weights.get(mood, 0) * val for mood, val in mood_detail.items())


def get_dominant_mood(mood_detail):
    """Return dominant mood as capitalized string or 'Neutral' if no dominant mood."""
    if not mood_detail:
        return "Neutral"
    dominant = max(mood_detail.items(), key=lambda x: x[1])
    return "Neutral" if dominant[1] == 0 else dominant[0].capitalize()


def classify_dream(text):
    """Classify dream into one of predefined categories."""
    text_lower = text.lower()

    nightmare_keywords = ["chase", "dark", "monster", "fall", "death", "attack", "fear"]
    symbolic_keywords = ["fly", "sea", "sky", "colors", "magic", "strange", "dream", "fantasy"]
    social_keywords = ["friend", "family", "love", "party", "talk", "relationship"]

    if any(word in text_lower for word in nightmare_keywords):
        return "Nightmare 😱"
    elif any(word in text_lower for word in symbolic_keywords):
        return "Symbolic 🌈"
    elif any(word in text_lower for word in social_keywords):
        return "Social/Emotional ❤️"
    else:
        return "Other 🌀"


def preprocess_texts(texts):
    """Clean texts by removing non-alphabetical characters and converting to lowercase."""
    return [re.sub(r'[^a-z\s]', '', txt.lower()) for txt in texts]


def cluster_dreams(texts, n_clusters=4):
    """Cluster dream texts into n_clusters using TF-IDF and KMeans."""
    if len(texts) < n_clusters:
        return [0] * len(texts)

    cleaned = preprocess_texts(texts)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(cleaned)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(X)


def highlight_keywords(text, keywords):
    """Highlight keywords in text by surrounding them with ** for markdown bold."""
    for kw in keywords:
        text = re.sub(f"(?i)({re.escape(kw)})", r"**[\1]**", text)
    return text


def alert_on_negative_trend(df):
    """Check if mood trend is declining this week compared to last week."""
    today = datetime.date.today()
    this_week = df[df['date'] >= pd.Timestamp(today - datetime.timedelta(days=7))]
    last_week = df[
        (df['date'] < pd.Timestamp(today - datetime.timedelta(days=7))) &
        (df['date'] >= pd.Timestamp(today - datetime.timedelta(days=14)))
    ]

    if len(this_week) < 3 or len(last_week) < 3:
        return False

    return this_week['mood_score'].mean() < last_week['mood_score'].mean() - 0.1


def plot_mood_heatmap(df):
    """Plot mood intensity heatmap for last 30 days."""
    if df.empty:
        st.info("No data to display heatmap.")
        return

    last_30 = df[df['date'] >= pd.Timestamp(datetime.date.today() - datetime.timedelta(days=30))]
    if last_30.empty:
        st.info("No dreams logged in the last 30 days for heatmap.")
        return

    moods = ["Joy", "Trust", "Surprise", "Sadness", "Fear", "Anger"]
    mood_map = {
        "joy": "Joy",
        "trust": "Trust",
        "surprise": "Surprise",
        "sadness": "Sadness",
        "fear": "Fear",
        "anger": "Anger"
    }

    dates = pd.date_range(start=last_30['date'].min(), end=last_30['date'].max())
    heat_data = pd.DataFrame(0, index=dates.strftime('%Y-%m-%d'), columns=moods)

    for _, row in last_30.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        mood_detail = row['mood_detail']

        if isinstance(mood_detail, str):
            mood_detail = json.loads(mood_detail.replace("'", '"'))

        for m_key, m_label in mood_map.items():
            heat_data.at[date_str, m_label] += mood_detail.get(m_key, 0)

    fig = px.imshow(
        heat_data.T,
        labels=dict(x="Date", y="Mood", color="Intensity"),
        x=heat_data.index,
        y=heat_data.columns,
        color_continuous_scale='RdBu',
        origin='lower',
        title="Mood Intensity Heatmap (Last 30 Days)"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_cluster_summary(df):
    """Plot bar chart of number of dreams per cluster."""
    cluster_counts = df['cluster'].value_counts().sort_index()
    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Number of Dreams'},
        title='Number of Dreams per Cluster'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_mood_radar(mood_avg):
    """Plot radar chart for average mood scores."""
    categories = list(mood_avg.keys())
    values = list(mood_avg.values())

    values += values[:1]      # Closing the radar chart loop
    categories += categories[:1]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Mood Profile'
            )
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Radar Mood Profile (Last 7 Days)"
    )
    st.plotly_chart(fig, use_container_width=True)


def generate_dream_analysis_text(dream, mood_detail, mood_label, dream_type):
    """Generate poetic and emotionally engaging dream analysis text."""
    joy = mood_detail.get("joy", 0)
    fear = mood_detail.get("fear", 0)
    anger = mood_detail.get("anger", 0)
    sadness = mood_detail.get("sadness", 0)
    trust = mood_detail.get("trust", 0)
    surprise = mood_detail.get("surprise", 0)

    lines = [
        f"🌌 Your dream unfolds as a **{dream_type}**, infused with a dominant feeling of **{mood_label}**."
    ]

    if joy > 0.5:
        lines.append("😊 A glow of joy shines through your dream — it may reflect warmth, love, or moments you treasure.")
    if sadness > 0.4:
        lines.append("💧 There's a tender sadness, whispering of inner reflection or unspoken emotions.")
    if fear > 0.3:
        lines.append("😰 Echoes of fear hint at hidden worries or uncertainties lurking beneath your surface.")
    if anger > 0.3:
        lines.append("😠 Anger pulses in the background — perhaps a sign of tension or things left unsaid.")
    if trust > 0.4:
        lines.append("🤝 A strong sense of trust fills your dream — this may symbolize healing bonds or inner peace.")
    if surprise > 0.3:
        lines.append("😲 Surprise appears like lightning — unexpected ideas or events may be shaping your path.")
    if all(v < 0.1 for v in mood_detail.values()):
        lines.append("💤 This dream feels calm and neutral — like a quiet night breeze across a still lake.")

    lines.extend([
        "\n📓 Try recording your dreams each day — patterns will emerge like constellations in the sky. 🪐",
        "🧠 Dreams aren’t random — they’re messages from within, full of meaning, memory, and mystery.",
        "💬 For deeper, personalized insight, chat with an AI dream assistant like [ChatGPT](https://chat.openai.com), "
        "or launch the in-app assistant we built just for you. 🧘‍♀️✨"
    ])

    return "\n\n".join(lines)


# --- Main App Interface ---

st.set_page_config(page_title="Moodream Pro Advanced", page_icon="🌙", layout="wide")
st.title("🌙 Moodream Pro Advanced — Deep Dream & Emotion Analytics")

df = load_data()

if not df.empty:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

tabs = st.tabs([
    "🌌 Log Dream",
    "📊 Mood Analytics",
    "🧠 Topic Clustering",
    "🗃️ Manage Dreams"
])


# --- Tab 1: Log Dream ---

with tabs[0]:
    st.header("📝 Log Today's Dream")

    dream_text = st.text_area(
        "Describe your dream vividly:",
        height=180,
        placeholder="Write your dream here..."
    )

    if st.button("💾 Analyze & Save"):
        if not dream_text.strip():
            st.warning("🚫 Please describe your dream before saving.")
        else:
            mood_detail = analyze_mood_multi(dream_text)
            dominant_mood = get_dominant_mood(mood_detail)
            mood_score = get_weighted_mood_score(mood_detail)
            dream_type = classify_dream(dream_text)
            today = pd.Timestamp(datetime.date.today())

            if ((df['date'] == today) & (df['dream'] == dream_text)).any():
                st.info("ℹ️ This dream has already been logged for today.")
            else:
                new_entry = {
                    "date": today.strftime('%Y-%m-%d'),
                    "dream": dream_text,
                    "mood_score": mood_score,
                    "mood_label": dominant_mood,
                    "type": dream_type,
                    "topics": [],
                    "mood_detail": mood_detail
                }
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                save_data(df)

                st.success(f"🌟 Dream saved! Dominant Mood: **{dominant_mood}**, Type: **{dream_type}**")

                chatgpt_url = f"https://chat.openai.com/?model=gpt-4&prompt={dream_text.replace(chr(10), '%0A')}"
                st.markdown(f"[💬 Ask ChatGPT about this dream](<{chatgpt_url}>)", unsafe_allow_html=True)

                st.write("🔍 Mood Intensity Breakdown:")
                for m, val in mood_detail.items():
                    st.progress(val, text=f"{m.capitalize()}: {val:.2f}")

                analysis_text = generate_dream_analysis_text(dream_text, mood_detail, dominant_mood, dream_type)
                st.markdown(f"### 🧙 Dream Interpretation & Insight\n{analysis_text}")

                if alert_on_negative_trend(df):
                    st.error("⚠️ Mood trend is declining this week compared to the last one!")


# --- Tab 2: Mood Analytics ---

with tabs[1]:
    st.header("📈 Mood Analytics Dashboard")

    if df.empty:
        st.info("📭 No dream data available.")
    else:
        plot_mood_heatmap(df)
        st.divider()

        st.subheader("📅 Daily Mood Trends")
        moods = ["joy", "fear", "anger", "sadness", "surprise", "trust"]
        dates = sorted(df['date'].dropna().unique())
        daily_moods = {m: [] for m in moods}

        for d in dates:
            day_rows = df[df['date'] == d]
            for m in moods:
                vals = []
                for _, row in day_rows.iterrows():
                    mood_detail = row['mood_detail']
                    if isinstance(mood_detail, str):
                        mood_detail = json.loads(mood_detail.replace("'", '"'))
                    vals.append(mood_detail.get(m, 0))
                daily_moods[m].append(np.mean(vals) if vals else 0)

        fig = go.Figure()
        for m in moods:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=daily_moods[m],
                    mode='lines+markers',
                    name=m.capitalize()
                )
            )

        fig.update_layout(
            title="📊 Average Mood Over Time",
            xaxis_title="Date",
            yaxis_title="Intensity"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🧭 Radar Mood Profile (Last 7 Days)")

        last_7 = df[df['date'] >= pd.Timestamp(datetime.date.today() - datetime.timedelta(days=7))]
        if not last_7.empty:
            agg = {}
            for m in moods:
                vals = []
                for _, row in last_7.iterrows():
                    mood_detail = row['mood_detail']
                    if isinstance(mood_detail, str):
                        mood_detail = json.loads(mood_detail.replace("'", '"'))
                    vals.append(mood_detail.get(m, 0))
                agg[m] = np.mean(vals) if vals else 0
            plot_mood_radar(agg)


# --- Tab 3: Topic Clustering ---

with tabs[2]:
    st.header("🔍 Topic Clustering & Dream Search")

    if df.empty:
        st.info("📭 No dreams to cluster or search.")
    else:
        n_clusters = st.slider("🔢 Number of Clusters", 2, 8, 4)
        df['cluster'] = cluster_dreams(df['dream'].tolist(), n_clusters=n_clusters)
        plot_cluster_summary(df)

        st.divider()
        st.subheader("🔎 Search Dreams by Keyword")
        search_term = st.text_input("Enter keyword(s):")
        if search_term:
            results = df[df['dream'].str.contains(search_term, case=False, na=False)]
            st.markdown(f"**Found {len(results)} result(s)**:")
            for _, row in results.iterrows():
                st.write(f"- {highlight_keywords(row['dream'], search_term.split())}")


# --- Tab 4: Manage Dreams ---

with tabs[3]:
    st.header("🧹 Manage & Edit Your Dreams")

    if df.empty:
        st.info("📭 No dreams recorded.")
    else:
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if not edited_df.equals(df):
            st.success("✅ Changes detected. Saving...")
            save_data(edited_df)
            df = edited_df

    st.divider()
    if st.button("🗑️ Clear All Dreams"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        df = pd.DataFrame(columns=['date', 'dream', 'mood_score', 'mood_label', 'type', 'topics', 'mood_detail'])
        st.warning("⚠️ All dream records cleared!")
