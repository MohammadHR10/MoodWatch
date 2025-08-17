# app/streamlit_demo.py
import os
import sys
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv
import json

# Add the project root to Python path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import with error handling
try:
    from app.llm_client import gemini_reply
    from app.au_payload import build_au_payload
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    LLM_ERROR = str(e)

# Demo data for when no CSV exists
DEMO_DATA = [
    {"ts": "2025-08-16 10:00:00", "AU01_r": 0.1, "AU04_r": 0.2, "AU12_r": 0.8, "AU06_r": 0.7, "AU25_r": 0.1, "AU26_r": 0.0, "AU45_c": 0.1, "valence_proxy": 0.6, "arousal_proxy": 0.2, "expr": "happy", "expr_score": 0.75},
    {"ts": "2025-08-16 10:01:00", "AU01_r": 0.3, "AU04_r": 0.6, "AU12_r": 0.2, "AU06_r": 0.1, "AU25_r": 0.2, "AU26_r": 0.1, "AU45_c": 0.3, "valence_proxy": -0.4, "arousal_proxy": 0.6, "expr": "sad", "expr_score": 0.65},
    {"ts": "2025-08-16 10:02:00", "AU01_r": 0.0, "AU04_r": 0.8, "AU12_r": 0.1, "AU06_r": 0.0, "AU25_r": 0.1, "AU26_r": 0.0, "AU45_c": 0.2, "valence_proxy": -0.7, "arousal_proxy": 0.3, "expr": "anger", "expr_score": 0.80},
    {"ts": "2025-08-16 10:03:00", "AU01_r": 0.5, "AU04_r": 0.3, "AU12_r": 0.4, "AU06_r": 0.3, "AU25_r": 0.6, "AU26_r": 0.4, "AU45_c": 0.4, "valence_proxy": 0.1, "arousal_proxy": 1.4, "expr": "surprise", "expr_score": 0.85},
    {"ts": "2025-08-16 10:04:00", "AU01_r": 0.1, "AU04_r": 0.1, "AU12_r": 0.9, "AU06_r": 0.8, "AU25_r": 0.1, "AU26_r": 0.0, "AU45_c": 0.1, "valence_proxy": 0.8, "arousal_proxy": 0.2, "expr": "happy", "expr_score": 0.90},
]

# -------------------------------------------------------------------
# App Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="MoodWatch - Facial Expression Analysis", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎭 MoodWatch - Facial Expression Analysis")
st.caption("Demo version - Real-time emotion detection and AI interpretation")

# -------------------------------------------------------------------
# Sidebar Info
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📊 About MoodWatch")
    st.info("""
    MoodWatch analyzes facial expressions using Action Units (AUs) to detect emotions in real-time.
    
    **Features:**
    - 🎥 Live camera emotion detection
    - 📈 Valence & arousal tracking  
    - 🤖 AI-powered interpretation
    - 📊 Historical emotion patterns
    """)
    
    st.header("🔧 Demo Mode")
    st.warning("""
    This is a demo version using sample data. 
    
    For live camera analysis, run locally:
    ```bash
    python -m app.camera_schedule
    streamlit run app/streamlit_app.py
    ```
    """)

# -------------------------------------------------------------------
# Load or create demo data
# -------------------------------------------------------------------
CSV_PATH = ROOT / "processed" / "session_summary.csv"

if CSV_PATH.exists():
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["ts"], engine="python", on_bad_lines="skip")
    except:
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

# Use demo data if no real data exists
if df.empty:
    st.info("📊 Using demo data - no live session data found")
    df = pd.DataFrame(DEMO_DATA)
    df['ts'] = pd.to_datetime(df['ts'])
else:
    st.success(f"📊 Loaded {len(df)} real emotion samples")

# Ensure required columns exist
for col in ["AU12_r", "AU04_r", "AU25_r", "AU26_r", "AU45_c", "valence_proxy", "arousal_proxy"]:
    if col not in df.columns:
        df[col] = 0.0

df = df.sort_values("ts").reset_index(drop=True)

# -------------------------------------------------------------------
# Data Overview
# -------------------------------------------------------------------
st.subheader("📈 Emotion Data Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Samples", len(df))
with col2:
    latest_emotion = df.iloc[-1]['expr'] if 'expr' in df.columns and len(df) > 0 else "N/A"
    st.metric("Latest Emotion", latest_emotion)
with col3:
    avg_valence = df['valence_proxy'].mean() if len(df) > 0 else 0
    st.metric("Avg Valence", f"{avg_valence:.2f}")
with col4:
    avg_arousal = df['arousal_proxy'].mean() if len(df) > 0 else 0
    st.metric("Avg Arousal", f"{avg_arousal:.2f}")

# Show recent data
st.dataframe(df.tail(5), use_container_width=True)

# -------------------------------------------------------------------
# Charts
# -------------------------------------------------------------------
if len(df) > 1:
    st.subheader("📊 Emotion Trends Over Time")
    
    # Valence & Arousal chart
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('ts:T', title='Time'),
        y=alt.Y('valence_proxy:Q', title='Valence (Positive/Negative)'),
        color=alt.value('blue'),
        tooltip=['ts:T', 'valence_proxy:Q', 'arousal_proxy:Q', 'expr:N']
    ).properties(height=200, title="Valence Over Time") + \
    alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('ts:T', title='Time'),
        y=alt.Y('arousal_proxy:Q', title='Arousal (Activation Level)'),
        color=alt.value('red'),
        tooltip=['ts:T', 'valence_proxy:Q', 'arousal_proxy:Q', 'expr:N']
    ).properties(height=200)
    
    st.altair_chart(chart.resolve_scale(y='independent'), use_container_width=True)
    
    # Expression distribution
    if 'expr' in df.columns:
        st.subheader("🎭 Expression Distribution")
        expr_counts = df['expr'].value_counts().reset_index()
        expr_counts.columns = ['Expression', 'Count']
        
        pie_chart = alt.Chart(expr_counts).mark_arc().encode(
            theta='Count:Q',
            color=alt.Color('Expression:N', scale=alt.Scale(scheme='category10')),
            tooltip=['Expression:N', 'Count:Q']
        ).properties(height=300)
        
        st.altair_chart(pie_chart, use_container_width=True)

# -------------------------------------------------------------------
# LLM Interpretation
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 AI Emotion Coach")

if not LLM_AVAILABLE:
    st.error(f"❌ LLM components not available: {LLM_ERROR}")
elif not os.getenv("GEMINI_API_KEY"):
    st.warning("⚠️ GEMINI_API_KEY not configured. Add your API key to use AI interpretation.")
    st.info("Get your API key from: https://makersuite.google.com/app/apikey")
else:
    st.success("✅ AI Coach ready")

    mode = st.radio("Analyze", ["Latest sample", "Last 10 samples"], horizontal=True)
    question = st.text_input("Ask your question", "What do my emotions tell me?")
    
    if st.button("🎯 Get AI Interpretation", type="primary"):
        if len(df) == 0:
            st.error("No emotion data available for analysis")
        else:
            with st.spinner("🤖 AI Coach is analyzing your emotions..."):
                # Create payload manually for demo
                if mode == "Latest sample":
                    latest = df.iloc[-1].to_dict()
                    payload = {
                        "_mode": "single_pulse",
                        "_ok": True,
                        "ts": str(latest["ts"]),
                        **{k: v for k, v in latest.items() if k.startswith("AU") or k in ["valence_proxy", "arousal_proxy", "expr", "expr_score"]}
                    }
                else:
                    window = df.tail(10)
                    means = {c: float(window[c].mean()) for c in window.columns if c.startswith("AU") or c in ["valence_proxy", "arousal_proxy"]}
                    payload = {
                        "_mode": "window_means",
                        "_ok": True,
                        "count": len(window),
                        "means": means
                    }
                
                try:
                    reply = gemini_reply(question, payload)
                    st.markdown("**🤖 AI Coach:** " + reply)
                    
                    with st.expander("📊 View emotion data sent to AI"):
                        st.json(payload)
                except Exception as e:
                    st.error(f"AI interpretation failed: {str(e)}")

# -------------------------------------------------------------------
# Action Units Reference
# -------------------------------------------------------------------
with st.expander("📚 Action Units (AU) Reference Guide"):
    st.markdown("""
    **Key Action Units for Emotion Detection:**
    
    | AU | Description | Emotion Hints |
    |---|---|---|
    | AU01 | Inner brow raise | Sadness, concern |
    | AU04 | Brow lowerer | Anger, concentration |
    | AU06 | Cheek raiser | Genuine joy |
    | AU12 | Lip corner puller | Smile, happiness |
    | AU15 | Lip corner depressor | Sadness |
    | AU25/26 | Lips part/jaw drop | Surprise, arousal |
    
    **Emotion Patterns:**
    - 😊 **Happy**: AU12 + AU06 (smile + cheek raise)
    - 😢 **Sad**: AU01 + AU04 + AU15 (inner brow + frown + down mouth)  
    - 😠 **Angry**: AU04 + AU07 + AU23 (frown + narrow eyes + tight lips)
    - 😮 **Surprised**: AU01 + AU02 + AU25/26 (raised brows + open mouth)
    """)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎭 MoodWatch - Facial Expression Analysis powered by OpenFace & Gemini AI</p>
    <p><small>For live emotion detection, run locally with camera access</small></p>
</div>
""", unsafe_allow_html=True)
