# app/streamlit_app.py
import os
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

# -------------------------------------------------------------------
# Paths & config
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "processed" / "session_summary.csv"

st.set_page_config(page_title="Facial expression changes", layout="centered")
st.title("Facial expression changes over time")

alt.data_transformers.disable_max_rows()

# -------------------------------------------------------------------
# Load data (be forgiving if you have older rows with different shape)
# -------------------------------------------------------------------
if not CSV_PATH.exists():
    st.warning("No session_summary.csv yet. Run the scheduler first.")
    st.stop()

# Older runs might have mismatched columns → skip bad lines
df = pd.read_csv(
    CSV_PATH,
    parse_dates=["ts"],
    engine="python",
    on_bad_lines="skip",
)

if df.empty:
    st.warning("session_summary.csv is empty after reading. Take a new pulse and refresh.")
    st.stop()

# Make sure essential columns exist (fill 0.0 for missing AUs)
for col in [
    "AU12_r", "AU04_r", "AU25_r", "AU26_r", "AU45_c",
    "dur_s", "frames"
]:
    if col not in df.columns:
        df[col] = 0.0

# Proxies (compute if missing)
if "valence_proxy" not in df.columns:
    df["valence_proxy"] = df["AU12_r"] - df["AU04_r"]

if "arousal_proxy" not in df.columns:
    df["arousal_proxy"] = df["AU25_r"] + df["AU26_r"] + df["AU45_c"]

# Sort by time & show the most recent few rows
df = df.sort_values("ts").reset_index(drop=True)
st.caption(f"Loaded **{len(df)}** pulses from: `{CSV_PATH}`")
st.dataframe(df.tail(10), use_container_width=True)

# Baseline = median of first K pulses (or just first row if fewer)
K = min(3, len(df))
baseline = df.head(K)[["valence_proxy", "arousal_proxy"]].median().to_dict()
df["valence_delta"] = df["valence_proxy"] - baseline.get("valence_proxy", 0.0)
df["arousal_delta"] = df["arousal_proxy"] - baseline.get("arousal_proxy", 0.0)

# Rolling means to smooth noise (5 pulses, at least 1)
df["valence_rm"] = df["valence_proxy"].rolling(5, min_periods=1).mean()
df["arousal_rm"] = df["arousal_proxy"].rolling(5, min_periods=1).mean()

# Convenience for small datasets: points instead of lines when <2 rows
is_tiny = len(df) < 2
mark_line = alt.MarkDef(type="line")
mark_point = alt.MarkDef(type="point")

# -------------------------------------------------------------------
# Valence & arousal (raw)
# -------------------------------------------------------------------
st.subheader("Valence & arousal (raw)")

raw_layer = alt.layer(
    alt.Chart(df).mark_point() if is_tiny else alt.Chart(df).mark_line()
        .encode(x="ts:T", y=alt.Y("valence_proxy:Q", title="valence (+smile −furrow)")),
    alt.Chart(df).mark_point() if is_tiny else alt.Chart(df).mark_line()
        .encode(x="ts:T", y=alt.Y("arousal_proxy:Q", title="arousal (mouth/blink)")),
).resolve_scale(y="independent").properties(height=280)

st.altair_chart(raw_layer, use_container_width=True)

# -------------------------------------------------------------------
# Valence & arousal (rolling mean)
# -------------------------------------------------------------------
st.subheader("Valence & arousal (rolling mean)")

smooth_layer = alt.layer(
    alt.Chart(df).mark_line().encode(x="ts:T", y=alt.Y("valence_rm:Q", title="valence (5-pulse avg)")),
    alt.Chart(df).mark_line().encode(x="ts:T", y=alt.Y("arousal_rm:Q", title="arousal (5-pulse avg)")),
).resolve_scale(y="independent").properties(height=280)

st.altair_chart(smooth_layer, use_container_width=True)

# -------------------------------------------------------------------
# Delta from baseline (how the session shifts vs. your starting state)
# -------------------------------------------------------------------
st.subheader("Change vs. baseline (first few pulses)")

delta_layer = alt.layer(
    alt.Chart(df).mark_line().encode(x="ts:T", y=alt.Y("valence_delta:Q", title="Δ valence")),
    alt.Chart(df).mark_line().encode(x="ts:T", y=alt.Y("arousal_delta:Q", title="Δ arousal")),
).resolve_scale(y="independent").properties(height=240)

st.altair_chart(delta_layer, use_container_width=True)

# -------------------------------------------------------------------
# Expression counts (if classifier columns exist)
# -------------------------------------------------------------------
if "expr" in df.columns:
    st.subheader("Detected expression (per pulse)")
    counts = df["expr"].value_counts().rename_axis("expr").reset_index(name="count")
    bar = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("expr:N", sort="-y"),
            y="count:Q",
            tooltip=["expr", "count"],
        )
        .properties(height=240)
    )
    st.altair_chart(bar, use_container_width=True)

# -------------------------------------------------------------------
# Download
# -------------------------------------------------------------------
st.download_button(
    "Download session_summary.csv",
    data=Path(CSV_PATH).read_bytes(),
    file_name="session_summary.csv",
    mime="text/csv",
)
