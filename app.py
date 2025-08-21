# app.py
# NFL Rosters Explorer — fixed CSV, team filter, and player search

import io
import pandas as pd
import numpy as np
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# ========================
# CONFIGURATION
# ========================
st.set_page_config(page_title="CFB Rosters Explorer", layout="wide")
st.title("🏈 CFB Rosters Explorer")
st.caption("Filter & explore the 2025 rosters with 2024 PFF stats")

# ========================
# COLUMNS TO DISPLAY
# ========================
DISPLAY_COLS = [
    "name_clean", "team", "Position", "pff_grades", "rating", "avg_depth_of_target", "yards_per_reception", "yprr",
    "grades_pass_route", "receptions", "yards", "route_rate",
    "wide_rate", "slot_rate", "elusive_rating"
]

# ========================
# LOAD DATA (HARDCODED CSV)
# ========================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("rosters_2025_w_2024_pff_stats.csv")
    # Keep only WR, TE, RB
    df = df[df["Position"].isin(["WR", "TE", "RB"])]
    # Filter to only desired columns (ignore missing gracefully)
    cols_to_use = [c for c in DISPLAY_COLS if c in df.columns]
    df = df[cols_to_use]
    return df

df = load_data()

# ========================
# PLAYER SEARCH (TOP OF PAGE)
# ========================
search_name = st.text_input("🔎 Search player (by name)", placeholder="Type a player name…")
filtered_df = df.copy()
if search_name:
    q = search_name.strip().lower()
    if "name_clean" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["name_clean"].astype(str).str.lower().str.contains(q, na=False)]

# ========================
# TEAM FILTER (SIDEBAR)
# ========================
teams = sorted(filtered_df["team"].dropna().unique()) if "team" in filtered_df.columns else []
selected_teams = st.sidebar.multiselect(
    "Select Teams",
    options=teams,
    default=teams  # Default = all teams selected
)
if selected_teams:
    filtered_df = filtered_df[filtered_df["team"].isin(selected_teams)]

# ========================
# DISPLAY RESULTS
# ========================
st.subheader("Filtered Rosters")
st.write(f"Showing **{len(filtered_df):,}** of **{len(df):,}** players")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# ========================
# DOWNLOAD CSV BUTTON
# ========================
@st.cache_data(show_spinner=False)
def to_csv_bytes(df_: pd.DataFrame) -> bytes:
    with io.StringIO() as buf:
        df_.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

csv_bytes = to_csv_bytes(filtered_df)
st.download_button(
    "⬇️ Download Filtered CSV",
    data=csv_bytes,
    file_name="filtered_rosters.csv",
    mime="text/csv",
)

