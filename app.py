# app.py
# Streamlit Data Filter App (dynamic filters for any tabular dataset)

import io
from typing import List, Tuple
import pandas as pd
import numpy as np
import streamlit as st
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
)

st.set_page_config(page_title="Data Filter App", layout="wide")
st.title("🔎 Data Explorer & Filter")
st.caption("Upload a CSV/Parquet or connect code to your data source. Filter by any column, then export the result.")

# ===============
# Data Loading
# ===============
@st.cache_data(show_spinner=False)
def load_file(upload) -> pd.DataFrame:
    if upload.name.lower().endswith(".parquet"):
        return pd.read_parquet(upload)
    # default to CSV
    return pd.read_csv(upload)

@st.cache_data(show_spinner=False)
def load_example() -> pd.DataFrame:
    # Small sample dataset if user doesn't upload anything
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=365, freq="D"),
        "category": np.random.choice(["A","B","C"], size=365, p=[0.5,0.3,0.2]),
        "name": np.random.choice(["alpha","bravo","charlie","delta"], size=365),
        "value": np.random.normal(loc=100, scale=15, size=365).round(2),
        "count": np.random.randint(0, 500, size=365)
    })
    return df

left, right = st.columns([3, 2])
with left:
    uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv","parquet"], help="File stays local to the session unless you deploy the app.")
with right:
    use_example = st.toggle("Use example dataset", value=uploaded is None)

if uploaded is not None:
    df = load_file(uploaded)
elif use_example:
    df = load_example()
else:
    st.info("👆 Upload a file or toggle the example dataset to begin.")
    st.stop()

# Coerce likely date columns (by name or dtype) for better filtering
for col in df.columns:
    if is_datetime64_any_dtype(df[col]):
        continue
    if any(k in str(col).lower() for k in ["date", "time", "timestamp", "dt"]):
        try:
            df[col] = pd.to_datetime(df[col], errors="ignore")
        except Exception:
            pass

st.subheader("Dataset Overview")
meta_cols = st.columns(5)
with meta_cols[0]:
    st.metric("Rows", f"{len(df):,}")
with meta_cols[1]:
    st.metric("Columns", f"{df.shape[1]:,}")
with meta_cols[2]:
    st.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
with meta_cols[3]:
    mem = df.memory_usage(deep=True).sum()
    st.metric("Memory", f"{mem/1_048_576:,.2f} MB")
with meta_cols[4]:
    st.metric("Dtypes", ", ".join(sorted(set(str(t) for t in df.dtypes))))

# ======================
# Dynamic Filter Builder
# ======================
st.sidebar.header("Filters")

@st.cache_data(show_spinner=False)
def get_unique_sorted(series: pd.Series) -> List:
    vals = series.dropna().unique().tolist()
    try:
        vals = sorted(vals)
    except Exception:
        # Mixed types may fail; fallback to string sort
        vals = sorted(vals, key=lambda x: str(x))
    return vals

filtered_df = df.copy()

# Global text search (optional)
with st.sidebar.expander("🔤 Global text search", expanded=False):
    search_cols = st.multiselect("Columns to search", options=list(filtered_df.columns), help="We will apply a case-insensitive contains() across selected columns.")
    query = st.text_input("Search term")
    if query and search_cols:
        mask = pd.Series(False, index=filtered_df.index)
        q = str(query).lower()
        for c in search_cols:
            try:
                mask = mask | filtered_df[c].astype(str).str.lower().str.contains(q, na=False)
            except Exception:
                pass
        filtered_df = filtered_df[mask]

# Column-wise filters
for col in filtered_df.columns:
    with st.sidebar.expander(f"Filter: {col}", expanded=False):
        col_data = filtered_df[col]
        if is_numeric_dtype(col_data):
            min_val, max_val = float(np.nanmin(col_data)), float(np.nanmax(col_data))
            step = (max_val - min_val) / 100 if max_val != min_val else 1.0
            sel_min, sel_max = st.slider(
                "Range", min_value=float(min_val), max_value=float(max_val), value=(float(min_val), float(max_val)), step=step
            )
            filtered_df = filtered_df[(filtered_df[col] >= sel_min) & (filtered_df[col] <= sel_max)]
        elif is_datetime64_any_dtype(col_data):
            min_date = pd.to_datetime(col_data.min())
            max_date = pd.to_datetime(col_data.max())
            start, end = st.date_input(
                "Date range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
            # Convert to full-day bounds
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            filtered_df = filtered_df[(pd.to_datetime(filtered_df[col]) >= start_ts) & (pd.to_datetime(filtered_df[col]) <= end_ts)]
        else:
            # Treat everything else as categorical/text
            opts = get_unique_sorted(col_data)
            # Show most common first for long lists
            top_n = st.number_input("Show top N by frequency (0 = all)", min_value=0, value=min(25, len(opts)))
            display_opts = opts
            if top_n and len(opts) > top_n:
                freq_order = (
                    filtered_df[col]
                    .value_counts(dropna=True)
                    .reset_index()
                    .rename(columns={"index": col, col: "freq"})
                )
                display_opts = freq_order[col].head(top_n).tolist()
            selected = st.multiselect("Select", options=display_opts, default=[], help="Leave empty to include all")
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

# ======================
# Results & Export
# ======================
st.subheader("Filtered Results")
st.write(f"Showing **{len(filtered_df):,}** of **{len(df):,}** rows")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
)

# Quick aggregations
with st.expander("📊 Quick aggregation", expanded=False):
    group_col = st.selectbox("Group by", options=[None] + list(filtered_df.columns), index=0)
    if group_col:
        num_cols = [c for c in filtered_df.columns if is_numeric_dtype(filtered_df[c])]
        agg_col = st.selectbox("Aggregate column", options=num_cols)
        agg_fn = st.selectbox("Aggregation", options=["sum","mean","median","min","max","count"])
        grouped = getattr(filtered_df.groupby(group_col)[agg_col], agg_fn)().reset_index()
        st.dataframe(grouped, use_container_width=True, hide_index=True)

# Download button
@st.cache_data(show_spinner=False)
def to_csv_bytes(df_: pd.DataFrame) -> bytes:
    with io.StringIO() as buf:
        df_.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

csv_bytes = to_csv_bytes(filtered_df)
st.download_button(
    "⬇️ Download filtered CSV",
    data=csv_bytes,
    file_name="filtered_data.csv",
    mime="text/csv",
)

st.caption("Built with ❤️ Streamlit. TIP: add @st.cache_data to heavy loaders/transformations for speed.")
