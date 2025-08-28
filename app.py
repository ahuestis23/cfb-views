import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================
# Config
# =========================
DATA_PATH = "model_input_data.csv"
N_CLUSTERS = {"WR": 4, "TE": 2, "RB": 2}
MIN_RECS_BY_POS = {"WR": 200, "TE": 120, "RB": 120}

REQUIRED_COLS = [
    "scheduled_date","player_name","position","team_market",
    "rec_rec","rec_yds","targets","grades_pass_route","avg_depth_of_target",
    "route_rate","slot_rate","wide_rate","yards_per_reception","yprr",
]

FEATURE_COLS = [
    "targets","avg_depth_of_target","grades_pass_route",
    "slot_rate","wide_rate","yprr","yards_per_reception","route_rate"
]

# =========================
# Helpers
# =========================
def prob_to_american(p: float) -> str:
    if p <= 0:  return "+Inf"
    if p >= 1:  return "-Inf"
    if np.isclose(p, 0.5): return "+100"
    return f"+{int(round((1 - p)/p * 100))}" if p < 0.5 else f"-{int(round(p/(1 - p) * 100))}"

def summarize(array: np.ndarray, yard_lines: List[float] = (40, 50, 60, 80, 100, 120)) -> Dict[str, float]:
    out = {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }
    for L in yard_lines:
        out[f"P(>{L})"] = float((array > L).mean())
    return out

def norm_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
         .str.lower()
         .str.replace(r"[^\w\s]", " ", regex=True)
         .str.replace(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

# =========================
# Artifacts
# =========================
@dataclass
class PositionArtifacts:
    scaler: StandardScaler
    kmeans: KMeans
    cluster_to_yards: Dict[int, np.ndarray]
    position_yards: np.ndarray

@dataclass
class Artifacts:
    pos_artifacts: Dict[str, PositionArtifacts]
    players_table: pd.DataFrame

# =========================
# Data loading & fitting
# =========================
@st.cache_data(show_spinner=False)
def load_model_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"model_input_data missing columns: {missing}")
    df["name_norm"] = norm_text(df["player_name"])
    df["team_norm"] = norm_text(df["team_market"])
    df["pos_norm"]  = df["position"].str.upper().str.strip()
    df = df[df["pos_norm"].isin(["WR","TE","RB"])].copy()
    return df

@st.cache_resource(show_spinner=True)
def fit_clusters_and_distributions(df: pd.DataFrame) -> Artifacts:
    # one row per player for features
    agg_map = {c: "max" for c in FEATURE_COLS}
    players = (
        df.groupby(["name_norm","team_norm","pos_norm"], as_index=False)
          .agg(agg_map)
    )
    first_cols = df.groupby(["name_norm","team_norm","pos_norm"], as_index=False).agg(
        player_name=("player_name","first"),
        team_market=("team_market","first"),
        position=("position","first"),
    )
    players = players.merge(first_cols, on=["name_norm","team_norm","pos_norm"], how="left")

    pos_artifacts: Dict[str, PositionArtifacts] = {}

    for pos in ["WR","TE","RB"]:
        ptab = players[players["pos_norm"] == pos].copy()
        if ptab.empty: continue
        X = ptab[FEATURE_COLS].copy().fillna(ptab[FEATURE_COLS].median(numeric_only=True))
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        k = N_CLUSTERS[pos]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        ptab["cluster"] = kmeans.fit_predict(Xs)
        players.loc[players["pos_norm"] == pos, "cluster"] = ptab["cluster"].values

        # reception-level join to build empirical per-catch yards
        recs = df[df["pos_norm"] == pos].merge(
            ptab[["name_norm","team_norm","cluster"]],
            on=["name_norm","team_norm"], how="left", validate="m:1"
        )
        recs = recs[pd.to_numeric(recs["rec_yds"], errors="coerce").notna()].copy()
        recs["rec_yds"] = recs["rec_yds"].astype(float)

        cluster_to_yards: Dict[int, np.ndarray] = {}
        position_yards = recs["rec_yds"].values

        for cid, g in recs.groupby("cluster"):
            arr = g["rec_yds"].values
            if len(arr) >= MIN_RECS_BY_POS.get(pos, 200):
                cluster_to_yards[int(cid)] = arr

        pos_artifacts[pos] = PositionArtifacts(
            scaler=scaler, kmeans=kmeans,
            cluster_to_yards=cluster_to_yards,
            position_yards=position_yards
        )

    players["cluster"] = players["cluster"].astype("Int64")
    return Artifacts(pos_artifacts=pos_artifacts, players_table=players)

# =========================
# Simulation
# =========================
def assign_player_cluster(artifacts: Artifacts, row: pd.Series) -> Tuple[str, Optional[int]]:
    pos = str(row["pos_norm"]).upper()
    PA = artifacts.pos_artifacts.get(pos)
    if PA is None:
        return pos, None
    feats = pd.DataFrame([row[FEATURE_COLS].to_dict()])
    feats = feats.fillna(artifacts.players_table.loc[
        artifacts.players_table["pos_norm"] == pos, FEATURE_COLS
    ].median(numeric_only=True).to_dict())
    Xs = PA.scaler.transform(feats)
    cid = int(PA.kmeans.predict(Xs)[0])
    return pos, cid

def simulate_player(
    artifacts: Artifacts,
    player_name: str,
    team_market: str,
    position: str,
    projected_rec: float,
    projected_rec_yds: Optional[float] = None,
    receptions_mode: str = "poisson",
    n_sims: int = 100_000,
    seed: int = 7
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:

    name_n = norm_text(pd.Series([player_name])).iloc[0]
    team_n = norm_text(pd.Series([team_market])).iloc[0]
    pos_n  = position.upper().strip()

    PT = artifacts.players_table
    PA = artifacts.pos_artifacts.get(pos_n)
    if PA is None:
        raise ValueError(f"No artifacts for position '{pos_n}'")

    cand = PT[(PT["name_norm"] == name_n) & (PT["team_norm"] == team_n) & (PT["pos_norm"] == pos_n)]
    if cand.empty:
        cand = PT[(PT["name_norm"] == name_n) & (PT["pos_norm"] == pos_n)]

    # Final fallback: position-only
    if cand.empty:
        yards_pool = PA.position_yards
        return _simulate_from_pool(yards_pool, projected_rec, projected_rec_yds, receptions_mode, n_sims, seed)

    prow = cand.iloc[0]
    pos, cid = assign_player_cluster(artifacts, prow)
    yards_pool = PA.cluster_to_yards.get(cid, PA.position_yards)
    return _simulate_from_pool(yards_pool, projected_rec, projected_rec_yds, receptions_mode, n_sims, seed)

def _simulate_from_pool(
    yards_pool: np.ndarray,
    projected_rec: float,
    projected_rec_yds: Optional[float],
    receptions_mode: str,
    n_sims: int,
    seed: int
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    if receptions_mode == "poisson":
        Ns = rng.poisson(lam=max(0.0, projected_rec), size=n_sims)
    elif receptions_mode == "fixed":
        Ns = np.full(n_sims, int(round(projected_rec)))
    elif receptions_mode == "binomial_fractional":
        base = int(np.floor(projected_rec)); frac = projected_rec - base
        Ns = base + rng.binomial(n=1, p=max(0.0, min(1.0, frac)), size=n_sims)
    else:
        raise ValueError("receptions_mode must be 'poisson', 'fixed', or 'binomial_fractional'.")

    draws_idx = rng.integers(0, len(yards_pool), size=int(Ns.sum()) if Ns.sum() > 0 else 1)
    catch_yards = yards_pool[draws_idx].astype(float)

    if projected_rec_yds is not None and projected_rec > 0:
        current_mean = float(np.mean(yards_pool))
        target_mean  = projected_rec_yds / projected_rec
        if current_mean > 1e-9:
            catch_yards *= (target_mean / current_mean)

    totals = np.zeros(n_sims, dtype=float)
    cursor = 0
    for i, n in enumerate(Ns):
        if n > 0:
            totals[i] = float(catch_yards[cursor:cursor+n].sum())
            cursor += n

    sims = pd.DataFrame({"receptions": Ns, "yards": totals})
    summary = {
        "receptions": summarize(sims["receptions"].values, yard_lines=[]),
        "yards": summarize(sims["yards"].values),
    }
    return sims, summary

def joint_prob_and_odds(sims: pd.DataFrame, rec_cond: str = ">= 5", yds_cond: str = ">= 60") -> Tuple[float, str]:
    import operator, re
    op_map = {">=": operator.ge, "<=": operator.le, "==": operator.eq, ">": operator.gt, "<": operator.lt}
    def parse(cond: str):
        s = cond.replace(" ", "")
        m = re.match(r"(>=|<=|==|>|<)(-?\d+(\.\d+)?)$", s)
        if not m: raise ValueError("Use forms like '>= 5', '< 60', '== 4'")
        return op_map[m.group(1)], float(m.group(2))
    rop, rv = parse(rec_cond); yop, yv = parse(yds_cond)
    mask = rop(sims["receptions"].values, rv) & yop(sims["yards"].values, yv)
    p = float(mask.mean())
    return p, prob_to_american(p)

# =========================
# UI
# =========================
st.set_page_config(page_title="CFB Receiving Sims", page_icon="🏈", layout="wide")
st.title("CFB Receiving Simulator")

with st.sidebar:
    st.header("Inputs")
    data = load_model_input(DATA_PATH)
    artifacts = fit_clusters_and_distributions(data)

    # ---------- helpers (local to sidebar) ----------
    def _coerce_float(s, default, minv=None, maxv=None):
        try:
            v = float(str(s).replace(",", "").strip())
        except Exception:
            st.warning(f"Could not parse a number from '{s}'. Using default {default}.")
            return default
        if minv is not None and v < minv: v = minv
        if maxv is not None and v > maxv: v = maxv
        return v

    def _parse_optional_float(s, minv=None, maxv=None):
        s = str(s).strip()
        if s == "":
            return None
        try:
            v = float(s.replace(",", ""))
        except Exception:
            st.warning("Could not parse optional yards; leaving blank.")
            return None
        if minv is not None and v < minv: v = minv
        if maxv is not None and v > maxv: v = maxv
        return v

    def _coerce_int(s, default, minv=None, maxv=None):
        try:
            v = int(float(str(s).replace(",", "").strip()))
        except Exception:
            st.warning(f"Could not parse an integer from '{s}'. Using default {default}.")
            return default
        if minv is not None and v < minv: v = minv
        if maxv is not None and v > maxv: v = maxv
        return v

    # ---------- mode / player pick ----------
    use_pos_only = st.toggle("Use custom / position-only mode", value=False)

    if not use_pos_only:
        players_table = artifacts.players_table.copy()
        players_table["label"] = (
            players_table["player_name"] + " — " +
            players_table["team_market"] + " (" +
            players_table["position"] + ")"
        )
        players_table = players_table.sort_values(["position","player_name"])

        selected_label = st.selectbox(
            "Select player",
            players_table["label"].tolist(),
            index=0
        )
        sel_row = players_table.loc[players_table["label"] == selected_label].iloc[0]
        sel_name = sel_row["player_name"]
        sel_team = sel_row["team_market"]
        sel_pos  = sel_row["position"]
    else:
        sel_pos = st.selectbox("Position", ["WR", "TE", "RB"], index=0)
        sel_name = "UNKNOWN"
        sel_team = "UNKNOWN"

    # ---------- projections (text inputs, no steppers) ----------
    proj_rec_raw = st.text_input("Projected receptions", value="4.5")
    proj_rec = _coerce_float(proj_rec_raw, default=4.5, minv=0.0, maxv=20.0)

    proj_yds_raw = st.text_input("Projected receiving yards (optional)", value="")
    proj_yds = _parse_optional_float(proj_yds_raw, minv=0.0, maxv=300.0)

    n_sims_raw = st.text_input("Simulations", value="100000")
    n_sims = _coerce_int(n_sims_raw, default=100000, minv=1000, maxv=500000)

    rec_mode = st.selectbox("Receptions mode", ["poisson","fixed","binomial_fractional"], index=0)

    # ---------- prop lines (text inputs, no steppers) ----------
    st.markdown("---")
    st.header("Prop Lines")

    rec_line_raw = st.text_input("Reception line (e.g., 3.5 → U3.5 / O3.5)", value="3.5")
    rec_line = _coerce_float(rec_line_raw, default=3.5, minv=0.0, maxv=20.0)

    yds_line_raw = st.text_input("Receiving yards line (e.g., 44.5)", value="44.5")
    yds_line = _coerce_float(yds_line_raw, default=44.5, minv=0.0, maxv=300.0)

    yard_thresholds_csv = st.text_input("Yard thresholds (for U(rec)&Y≥X)", value="50,60,70,80,90")

    max_k_raw = st.text_input("Max receptions k for k+ rec & U(yds)", value="7")
    max_k = _coerce_int(max_k_raw, default=7, minv=1, maxv=20)

    run = st.button("Run simulation")

st.caption("Default joint queries shown: U4.5 receptions & 50/60/70/80/90/100+ yards (i.e., R<=4 & Y>=X).")

if run:
    if use_pos_only:
        st.info(f"Position-only mode: using {sel_pos} pooled distribution (no player record required).")
    sims, summary = simulate_player(
        artifacts,
        player_name=sel_name,
        team_market=sel_team,
        position=sel_pos,
        projected_rec=proj_rec,
        projected_rec_yds=proj_yds,
        n_sims=int(n_sims),
        receptions_mode=rec_mode,
        seed=42
    )

    # Summaries
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Receptions summary")
        st.json(summary["receptions"])
    with c2:
        st.subheader("Yards summary")
        st.json(summary["yards"])

    # Cluster info
    PT = artifacts.players_table
    prow = PT[(PT["player_name"] == sel_name) & (PT["team_market"] == sel_team) & (PT["position"] == sel_pos)]
    if prow.empty:
        st.warning("Player not found in players_table. Using position-wide fallback.")
        pos = sel_pos.upper()
        PA = artifacts.pos_artifacts.get(pos)
        kept = set(PA.cluster_to_yards.keys()) if PA else set()
        st.write(f"Kept clusters for {pos}: {sorted(kept)}")
    else:
        cid = int(prow.iloc[0]["cluster"]) if pd.notna(prow.iloc[0]["cluster"]) else None
        pos = prow.iloc[0]["position"]
        st.info(f"Position: {pos}  •  Cluster: {cid if cid is not None else 'N/A (fallback)'}")

    # ---------------------------
    # Block A: U(rec_line) & Y>= thresholds
    # ---------------------------
    try:
        y_thresholds = [float(x.strip()) for x in yard_thresholds_csv.split(",") if x.strip()]
    except Exception as e:
        st.error(f"Could not parse yard thresholds: {e}")
        y_thresholds = []

    # Under X.5 receptions == <= floor(X.5)
    # If the line is an integer like 4.0, U4.0 is <= 4 by convention; adjust if you prefer otherwise
    rec_under_val = int(np.floor(rec_line))
    rec_under_cond = f"<= {rec_under_val}"

    rows_a = []
    for y in y_thresholds:
        p, odds = joint_prob_and_odds(sims, rec_cond=rec_under_cond, yds_cond=f">= {y}")
        rows_a.append({
            "query": f"U{rec_line} & {int(y)}+ yds",
            "rec_condition": rec_under_cond,
            "yds_condition": f">= {int(y)}",
            "prob": round(p, 4),
            "american_odds": odds
        })

    st.subheader(f"Joint: U{rec_line} receptions & Y≥X")
    if rows_a:
        st.dataframe(pd.DataFrame(rows_a), use_container_width=True)
    else:
        st.info("Enter thresholds (e.g., 50,60,70,80,90) to see this table.")

    # ---------------------------
    # Block B: k+ receptions & U(yds_line)
    # ---------------------------
    yds_under_cond = f"<= {yds_line}"
    rows_b = []
    for k in range(1, int(max_k) + 1):
        p, odds = joint_prob_and_odds(sims, rec_cond=f">= {k}", yds_cond=yds_under_cond)
        rows_b.append({
            "query": f"{k}+ rec & U{yds_line} yds",
            "rec_condition": f">= {k}",
            "yds_condition": yds_under_cond,
            "prob": round(p, 4),
            "american_odds": odds
        })

    st.subheader(f"Joint: k+ receptions (k=1..{int(max_k)}) & U{yds_line} yds")
    st.dataframe(pd.DataFrame(rows_b), use_container_width=True)
