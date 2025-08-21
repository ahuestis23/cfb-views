import io
import pandas as pd
import numpy as np
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# ========================
# CONFIGURATION
# ========================
st.set_page_config(page_title="NFL Rosters Explorer", layout="wide")
st.title("🏈 NFL Rosters Explorer")
st.caption("Filter & explore the 2025 rosters with 2024 PFF stats")

# ========================
# LOAD DATA (HARDCODED CSV)
# ========================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("rosters_2025_w_2024_pff_stats.csv")
    # Keep only WR, TE, RB
    df = df[df["Position"].isin(["WR", "TE", "RB"])]
    return df

df = load_data()

# ========================
# TEAM FILTER
# ========================
teams = sorted(df["team"].dropna().unique())
selected_teams = st.sidebar.multiselect(
    "Select Teams", 
    options=teams,
    default=teams  # Default = all teams selected
)

filtered_df = df[df["team"].isin(selected_teams)]

# ========================
# DYNAMIC FILTERING FOR OTHER COLUMNS
# ========================
st.sidebar.header("Additional Filters")

for col in filtered_df.columns:
    if col in ["team", "Position"]:
        continue  # already handled

    with st.sidebar.expander(f"Filter: {col}", expanded=False):
        col_data = filtered_df[col]
        # Numeric slider
        if is_numeric_dtype(col_data):
            min_val, max_val = float(np.nanmin(col_data)), float(np.nanmax(col_data))
            step = (max_val - min_val) / 100 if max_val != min_val else 1.0
            sel_min, sel_max = st.slider(
                "Range", 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=(float(min_val), float(max_val)), 
                step=step
            )
            filtered_df = filtered_df[(filtered_df[col] >= sel_min) & (filtered_df[col] <= sel_max)]

        # Datetime filter
        elif is_datetime64_any_dtype(col_data):
            min_date = pd.to_datetime(col_data.min())
            max_date = pd.to_datetime(col_data.max())
            start, end = st.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            filtered_df = filtered_df[
                (pd.to_datetime(filtered_df[col]) >= start_ts) & (pd.to_datetime(filtered_df[col]) <= end_ts)
            ]

        # Categorical multiselect
        else:
            opts = filtered_df[col].dropna().unique().tolist()
            opts = sorted(opts, key=lambda x: str(x))
            selected = st.multiselect("Select", options=opts, default=opts)
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

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

