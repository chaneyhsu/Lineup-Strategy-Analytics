# uconn_app.py
# Streamlit — UConn roster: radar (12 axes), projections, rotation planner
# Fully English UI; robust column aliasing for FT/2P/3P; minutes step=1

from pathlib import Path
from typing import Optional, List, Dict
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="UConn Roster — Offense Fit & Projection", layout="wide")

# ----------------------------- helpers -----------------------------
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def only_num(x):
    if pd.isna(x): return np.nan
    s = str(x).replace(",", "")
    m = NUM_RE.search(s)
    return float(m.group()) if m else np.nan

def canonicalize_name(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def pick_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    if df.empty: return None
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    want = [norm(a) for a in aliases]
    for c in df.columns:
        nc = norm(c)
        if any(w in nc for w in want):
            return c
    return None

def normalize_pct_series(s: pd.Series) -> pd.Series:
    t = pd.to_numeric(s, errors="coerce")
    if t.dropna().between(0, 1).mean() > 0.6:
        t = (t * 100.0).round(1)
    else:
        t = t.round(1)
    return t

def neighbor_pct(df: pd.DataFrame, count_aliases: List[str]) -> Optional[pd.Series]:
    c = pick_col(df, count_aliases)
    if c is None: return None
    idx = df.columns.get_loc(c)
    if idx + 1 >= len(df.columns): return None
    cand = df.columns[idx + 1]
    s = pd.to_numeric(df[cand], errors="coerce")
    return normalize_pct_series(s)

def percentile(col: pd.Series) -> pd.Series:
    s = pd.to_numeric(col, errors="coerce")
    return (s.rank(pct=True, method="average") * 100).round(1)

def zscore(col: pd.Series) -> pd.Series:
    s = pd.to_numeric(col, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def team_val(team_df: pd.DataFrame, label_sub: str, col: str = "offense"):
    if team_df.empty: return np.nan
    if "category" not in team_df.columns or col not in team_df.columns: return np.nan
    m = team_df["category"].astype(str).str.contains(label_sub, case=False, na=False)
    if m.any():
        return pd.Series(team_df.loc[m, col]).apply(only_num).squeeze()
    return np.nan

def add_fit_score(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Score = 45% shooting + 35% decision + 20% usage."""
    df = df.copy()
    shoot_cols = [c for c in ["efg_pctl","3p_pct_pctl","ft_pct_pctl","ts_pctl"] if c in df.columns]
    decision_cols = [c for c in ["ast%_pctl","tov_inv_pctl"] if c in df.columns]
    shoot = df[shoot_cols].mean(axis=1) if shoot_cols else pd.Series(50, index=df.index)
    decision = df[decision_cols].mean(axis=1) if decision_cols else pd.Series(50, index=df.index)
    usage = df["usg_pctl"] if "usg_pctl" in df.columns else pd.Series(50, index=df.index)
    df["fit_score_uconn_0_100"] = (0.45*shoot + 0.35*decision + 0.20*usage).clip(0,100).round(1)
    return df

def lineup_proj_ortg(
    df: pd.DataFrame,
    players: List[str],
    three_pref: float,
    afgm_pref: float,
    tov_pref: float
) -> float:
    sub = df[df["Player"].isin(players)].copy()
    if sub.empty:
        return np.nan

    # 需要用到的欄位（bonus 用得到）
    need = ["efg_pctl", "3p_pct_pctl", "ast%_pctl", "tov_inv_pctl", "usg_pctl", "proj_ortg_uconn"]
    if not all(n in sub.columns for n in need):
        # 若欄缺，至少回傳 proj_ortg_uconn 的平均，避免報錯
        return pd.to_numeric(sub.get("proj_ortg_uconn", pd.Series(np.nan))).mean()

    # --------- 核心：使用率加權平均 ---------
    ortg = pd.to_numeric(sub["proj_ortg_uconn"], errors="coerce")

    # 權重優先順序：USG / usg / usg_pct / (usg_pctl/100)；權重須為非負
    if "USG" in sub.columns:
        w = pd.to_numeric(sub["USG"], errors="coerce").clip(lower=0)
    elif "usg" in sub.columns:
        w = pd.to_numeric(sub["usg"], errors="coerce").clip(lower=0)
    elif "usg_pct" in sub.columns:
        w = pd.to_numeric(sub["usg_pct"], errors="coerce").clip(lower=0)
    else:
        # 沒有實際 USG，就用百分位近似（0~1）
        w = pd.to_numeric(sub["usg_pctl"], errors="coerce").clip(lower=0) / 100.0

    valid = ortg.notna() & w.notna() & (w > 0)
    if valid.any():
        core = float(np.average(ortg[valid], weights=w[valid]))
    else:
        # 權重全缺或全 0，就退回簡單平均
        core = float(ortg.mean())

    # --------- bonus（維持原邏輯）---------
    spacing = sub[["efg_pctl", "3p_pct_pctl"]].mean(axis=1).mean()
    decision = sub[["ast%_pctl", "tov_inv_pctl"]].mean(axis=1).mean()
    usage_std = sub["usg_pctl"].std(ddof=0) if sub["usg_pctl"].notna().sum() >= 2 else 0
    bonus = 0.06 * (spacing - 50) + 0.05 * (decision - 50) - 0.03 * max(0.0, usage_std - 15)
    bonus = float(np.clip(bonus, -4, 4))

    return round(core + bonus, 1)


# ----------------------------- defaults -----------------------------
# ----------------------------- defaults -----------------------------
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent  # 程式所在資料夾

DEFAULTS = {
    "uconn_roster": [
        APP_DIR / "uconn_roster_2024-25_stats.csv",
        APP_DIR / "uconn_roster_clean.csv",
        APP_DIR / "uconn_roster_clean .csv",
    ],
    "uconn_team": [APP_DIR / "uconn_team_summary_2024-25.csv"],
    "ncaa_all": [APP_DIR / "ncaa_roster_stats_2024-25_all_positions.csv"],
}

NCAA_DEFAULTS = {"three_rate": 39.0, "afgm": 52.0, "tov": 17.2}


def load_first(paths: List[Path]):
    for p in paths:
        if isinstance(p, Path) and p.exists():
            return p
    return None

def read_csv_any(p: Optional[Path]) -> pd.DataFrame:
    if p is None: return pd.DataFrame()
    return pd.read_csv(p)

def normalize_player_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    for k in ["player","player_name","player name","姓名","球員"]:
        if k in low:
            df = df.rename(columns={low[k]: "Player"})
            break
    return df

# ----------------------------- uploads -----------------------------
st.sidebar.header("Data Sources")
up_uconn_roster = st.sidebar.file_uploader("UConn roster CSV ('Player' required)", type=["csv"])
up_uconn_team   = st.sidebar.file_uploader("UConn team summary CSV", type=["csv"])
up_ncaa_all     = st.sidebar.file_uploader("NCAA all players CSV (optional)", type=["csv"])

uconn_roster_path = up_uconn_roster or load_first(DEFAULTS["uconn_roster"])
uconn_team_path   = up_uconn_team   or load_first(DEFAULTS["uconn_team"])
ncaa_all_path     = up_ncaa_all     or load_first(DEFAULTS["ncaa_all"])

@st.cache_data(show_spinner=False)
def _load(u_p, t_p, n_p):
    u = normalize_player_column(read_csv_any(u_p))
    t = read_csv_any(t_p)
    if not t.empty:
        t.columns = [c.strip().lower() for c in t.columns]
    n = normalize_player_column(read_csv_any(n_p))
    return u, t, n

uconn_raw, team_raw, ncaa_raw = _load(uconn_roster_path, uconn_team_path, ncaa_all_path)

if uconn_raw.empty or team_raw.empty:
    st.warning("Please provide UConn roster + team summary CSVs in the sidebar.")
    st.stop()

# ----------------------------- team style baselines -----------------------------
adj_off  = team_val(team_raw, "adj. efficiency", "offense")
three_rt = team_val(team_raw, "3pa/fga", "offense")
afgm_rt  = team_val(team_raw, "a/fgm", "offense")
tov_rt   = team_val(team_raw, "turnover", "offense")

three_pref = (three_rt if not pd.isna(three_rt) else NCAA_DEFAULTS["three_rate"]) - NCAA_DEFAULTS["three_rate"]
afgm_pref  = (afgm_rt  if not pd.isna(afgm_rt)  else NCAA_DEFAULTS["afgm"])      - NCAA_DEFAULTS["afgm"]
tov_pref   = (tov_rt   if not pd.isna(tov_rt)   else NCAA_DEFAULTS["tov"])       - NCAA_DEFAULTS["tov"]

# ----------------------------- NCAA table (robust parsing) -----------------------------
def build_ncaa_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out: Dict[str, pd.Series] = {}
    if "Player" in df.columns:
        out["Player"] = df["Player"]

    team_col = pick_col(df, ["team"])
    if team_col: out["Team"] = df[team_col]
    role_col = pick_col(df, ["role","Role","position","pos"])
    if role_col: out["role"] = df[role_col]

    g_col = pick_col(df, ["g","games"])
    if g_col: out["games"] = pd.to_numeric(df[g_col], errors="coerce")
    min_col = pick_col(df, ["min%","min pct","min_pct","%min","%min.","%Min"])
    if min_col: out["min_pct"] = pd.to_numeric(df[min_col], errors="coerce")

    for name, aliases in [
        ("ortg", ["ortg","off rtg","offensive rating"]),
        ("efg",  ["efg","eFG%","effective fg%","effective field goal%"]),
        ("ts",   ["ts","ts%","true shooting","true shooting%"]),
        ("usg",  ["usg","usg%","usage%","usage"]),
        ("ast%", ["ast%","assist%","ast pct","assist pct","ast rate","assist rate"]),
        ("to%",  ["to%","tov%","turnover%","turnover rate","to rate"]),
        ("or%",  ["or%","off reb%","offensive reb%"]),
        ("dr%",  ["dr%","def reb%","defensive reb%"]),
        ("stl%", ["stl%","steal%","steal rate"]),
        ("blk%", ["blk%","block%","block rate"]),
    ]:
        c = pick_col(df, aliases)
        if c: out[name] = pd.to_numeric(df[c], errors="coerce")

    three = None
    c = pick_col(df, ["3p%","3pt%","3p pct","three%","3P-Pct","3P Pct","3P_pct"])
    if c: three = normalize_pct_series(df[c])
    else: three = neighbor_pct(df, ["3pm-a","3pm_a","3pm–a","3pmA","3PM-A","3P","3pt"])
    if three is not None: out["3p_pct"] = three

    ft = None
    c = pick_col(df, ["ft%","ft pct","free throw%","FT-Pct","Ft-Pct","FT Pct","FT_pct"])
    if c: ft = normalize_pct_series(df[c])
    else: ft = neighbor_pct(df, ["ftm-a","ftm_a","ftm–a","ftmA","FTM-A","FT"])
    if ft is not None: out["ft_pct"] = ft

    two = None
    c = pick_col(df, ["2p%","2pt%","2p pct","2P-Pct","2P Pct","2P_pct"])
    if c: two = normalize_pct_series(df[c])
    else: two = neighbor_pct(df, ["2pm-a","2pm_a","2pm–a","2pmA","2PM-A","2P"])
    if two is not None: out["2p_pct"] = two

    ncaa = pd.DataFrame(out)

    for c in ["ortg","efg","ts","3p_pct","2p_pct","ft_pct","usg","ast%","to%","min_pct","or%","dr%","stl%","blk%"]:
        if c in ncaa.columns: ncaa[f"{c}_pctl"] = percentile(ncaa[c])
    if "to%_pctl" in ncaa.columns:
        ncaa["tov_inv_pctl"] = (100 - ncaa["to%_pctl"]).round(1)

    shoot_index = zscore(pd.to_numeric(ncaa.get("efg", ncaa.get("3p_pct", pd.Series(np.nan))), errors="coerce"))
    playmake_index = zscore(pd.to_numeric(ncaa.get("ast%", pd.Series(np.nan)), errors="coerce"))
    tov_index = zscore(pd.to_numeric(-ncaa.get("to%", pd.Series(np.nan)), errors="coerce"))

    k_three, k_ast, k_tov = 2.0, 1.5, 2.0
    ortg_base = pd.to_numeric(ncaa.get("ortg", pd.Series(np.nan)), errors="coerce")
    ncaa["proj_ortg_uconn"] = (
        ortg_base
        + k_three*((three_pref/10.0) if not pd.isna(three_pref) else 0.0)*shoot_index
        + k_ast*((afgm_pref/10.0)  if not pd.isna(afgm_pref)  else 0.0)*playmake_index
        + k_tov*((-tov_pref/10.0)  if not pd.isna(tov_pref)   else 0.0)*tov_index
    ).round(1)
    ncaa["delta_ortg_uconn"] = (ncaa["proj_ortg_uconn"] - ortg_base).round(1)
    return ncaa

ncaa = build_ncaa_table(ncaa_raw)

# ----------------------------- merge NCAA to UConn roster -----------------------------
u = uconn_raw.copy()
if "Player" not in u.columns:
    st.error("UConn roster file must include a 'Player' column.")
    st.stop()

rename_map = {
    "ORtg":"ortg","eFG":"efg","TS":"ts","USG%":"usg","AST%":"ast%","TOV%":"to%",
    "3P%":"3p_pct","FT%":"ft_pct","2P%":"2p_pct","Min%":"min_pct","Role":"role","Roles":"role","ROLE":"role"
}
for k,v in rename_map.items():
    if k in u.columns and v not in u.columns:
        u[v] = pd.to_numeric(u[k], errors="coerce") if v != "role" else u[k]

for src, dst in [("FT-Pct","ft_pct"), ("2P-Pct","2p_pct"), ("3P-Pct","3p_pct")]:
    if src in u.columns and dst not in u.columns:
        u[dst] = normalize_pct_series(u[src])

for c in ["ortg","efg","ts","3p_pct","2p_pct","ft_pct","usg","ast%","to%","min_pct","or%","dr%","stl%","blk%"]:
    if c in u.columns and f"{c}_pctl" not in u.columns:
        u[f"{c}_pctl"] = percentile(u[c])
if "to%_pctl" in u.columns and "tov_inv_pctl" not in u.columns:
    u["tov_inv_pctl"] = (100 - u["to%_pctl"]).round(1)

u["player_key"] = u["Player"].astype(str).apply(canonicalize_name)
if not ncaa.empty:
    tmp = ncaa.copy()
    tmp["player_key"] = tmp["Player"].astype(str).apply(canonicalize_name)
    u_join = u.merge(tmp.drop_duplicates("player_key"),
                     on="player_key", how="left", suffixes=("","_ncaa"))
else:
    u_join = u.copy()

def _add_basic_box(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    alias_map = {
        "ppg": ["ppg","pts/g","points per game","pts per game","pts","points","PTS","PTS/G"],
        "rpg": ["rpg","reb/g","rebs/g","rebounds per game","trb","reb","rebs","TRB","TRB/G"],
        "apg": ["apg","ast/g","assists per game","ast","assists","AST","AST/G"],
        "bpg": ["bpg","blk/g","blocks per game","BLK/G","blocks"],
        "spg": ["spg","stl/g","steals per game","STL/G","steals"],
    }
    for new, aliases in alias_map.items():
        c = pick_col(df, aliases)
        if c and new not in df.columns:
            df[new] = pd.to_numeric(df[c], errors="coerce")
    if "bpg" not in df.columns:
        c_blk_pct = pick_col(df, ["blk%","block%","blk pct","block pct"])
        if c_blk_pct and "blk_pct" not in df.columns:
            df["blk_pct"] = pd.to_numeric(df[c_blk_pct], errors="coerce")
    if "spg" not in df.columns:
        c_stl_pct = pick_col(df, ["stl%","steal%","stl pct","steal pct"])
        if c_stl_pct and "stl_pct" not in df.columns:
            df["stl_pct"] = pd.to_numeric(df[c_stl_pct], errors="coerce")
    return df

u_join = _add_basic_box(u_join)

if "proj_ortg_uconn" not in u_join.columns and "ortg" in u_join.columns:
    u_join["proj_ortg_uconn"] = pd.to_numeric(u_join["ortg"], errors="coerce")
    u_join["delta_ortg_uconn"] = 0.0

u_join = add_fit_score(u_join)

PCTL_KEYS = ["efg","ts","2p_pct","3p_pct","ft_pct","usg","ast%","to%","dr%","or%","stl%","blk%"]
for k in PCTL_KEYS:
    s = None
    if f"{k}_pctl_ncaa" in u_join.columns: s = u_join[f"{k}_pctl_ncaa"]
    elif f"{k}_pctl" in u_join.columns:    s = u_join[f"{k}_pctl"]
    elif k in u_join.columns:              s = percentile(u_join[k])
    if s is not None:
        u_join[f"{k}_pctl_best"] = pd.to_numeric(s, errors="coerce")
if "to%_pctl_best" in u_join.columns:
    u_join["tov_inv_pctl_best"] = (100 - u_join["to%_pctl_best"]).round(1)

role_col = None
for cand in ["role","Role","position","Position","pos","Pos"]:
    if cand in u_join.columns:
        role_col = cand; break

# ----------------------------- KPIs & controls -----------------------------
st.title("UConn Offense Fit & Projection (2024–25)")

# --- Header KPIs ---
BETA_3PA_PER_PCTPOINT = 0.49  # 每 +1% 三分出手比例 ≈ +0.49 AdjO 分（單變量迴歸估計）
GAMMA_AFGM = 30.55  # 每 +1.00 的 A/FGM（AST/FGM）≈ +30.55 AdjO 分（迴歸估計）

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("UConn AdjO (KenPom)", f"{adj_off:.1f}" if not pd.isna(adj_off) else "—")

with c2:
    # 與 D-I 平均的 3PA/FGA 差異（百分點）
    delta_three_pctpt = (three_rt - NCAA_DEFAULTS["three_rate"]) if not pd.isna(three_rt) else None
    # 轉為對 AdjO 的估計影響（使用 β = 0.49 / 每 1%）
    three_adj_impact = (delta_three_pctpt * BETA_3PA_PER_PCTPOINT) if delta_three_pctpt is not None else None
    st.metric(
        "3PA/FGA impact (β=0.49)",
        f"{three_adj_impact:+.1f} pts" if three_adj_impact is not None else "—",
        help="Estimated AdjO change = (UConn 3PA/FGA − D-I avg) × 0.49"
    )

with c3:
    # A/FGM uplift using γ (per 1.00 in A/FGM)
    # 資料中的 afgm_rt 與 NCAA_DEFAULTS['afgm'] 是「百分比值」（例如 69 表 0.69）。
    # 先換成比值差距，再乘上 γ。
    delta_afgm_ratio = ((afgm_rt - NCAA_DEFAULTS['afgm'])/100.0) if not pd.isna(afgm_rt) else None
    afgm_uplift = (delta_afgm_ratio * GAMMA_AFGM) if delta_afgm_ratio is not None else None
    st.metric(
        'A/FGM vs D-I',
        f"{afgm_uplift:+.1f} pts" if afgm_uplift is not None else '—',
        help='ΔAdjO = γ × (A/FGM_team − A/FGM_D-I); γ=30.55 (per 1.00). A/FGM here uses AST/FGM.'
    )


with st.expander("Filters (UConn Rankings)", expanded=True):
    role_opts = ["All"]
    if role_col: role_opts += sorted(u_join[role_col].dropna().astype(str).unique())
    role_pick = st.selectbox("Role", role_opts, index=0)

    min_minpct = 10 if "min_pct" in u_join.columns else 0
    if "min_pct" in u_join.columns:
        min_minpct = st.slider("Min% (minutes share) ≥", 0, 80, 10, step=5)

    sort_candidates = [c for c in ["ortg",
                                   "efg","ts","3p_pct","2p_pct","ft_pct","usg"] if c in u_join.columns]
    sort_by = st.selectbox("Sort by", sort_candidates, index=0 if sort_candidates else 0)

    top_k = st.slider("Show Top K", 5, 50, 15, step=5)
    search = st.text_input("Search player name (contains)")

    # ---------- NCAA School -> Player quick picker ----------
    if ncaa.empty:
        ncaa_school = "— None —"
        ncaa_player = None
        st.caption("Upload NCAA players CSV to enable NCAA school/player picker.")
    else:
        ncaa_school_opts = ["— None —"] + sorted(ncaa["Team"].dropna().astype(str).unique())
        ncaa_school = st.selectbox("NCAA School (optional)", ncaa_school_opts, index=0, help="Type to search")
        ncaa_player = None
        if ncaa_school != "— None —":
            pool_tmp = ncaa[ncaa["Team"].astype(str) == ncaa_school]
            ply_opts = sorted(pool_tmp["Player"].dropna().astype(str).unique())
            if ply_opts:
                ncaa_player = st.selectbox("NCAA Player", ply_opts, help="Type to search")
            else:
                st.info("No players under this school.")

    st.session_state["ncaa_school_pick"] = ncaa_school
    st.session_state["ncaa_player_pick"] = ncaa_player

# ----- Rankings table (DEFAULT = NCAA All; filter to school if chosen; fallback UConn if NCAA empty) -----
filtered_title = "Rankings — Filtered Players"
ncaa_school_pick = st.session_state.get("ncaa_school_pick", "— None —")

if not ncaa.empty:
    pool = ncaa.copy()
    if ncaa_school_pick and ncaa_school_pick != "— None —":
        pool = pool[pool["Team"].astype(str) == ncaa_school_pick]
        title_suffix = f"NCAA ({ncaa_school_pick})"
        dl_name = f"rankings_ncaa_{ncaa_school_pick}.csv"
    else:
        title_suffix = "NCAA (All schools)"
        dl_name = "rankings_ncaa_all.csv"

    # role filter that works for either column name
    if role_pick != "All":
        if "role" in pool.columns:
            pool = pool[pool["role"].astype(str) == role_pick]
        elif role_col and role_col in pool.columns:
            pool = pool[pool[role_col].astype(str) == role_pick]

    if "min_pct" in pool.columns:
        pool = pool[pool["min_pct"].fillna(0) >= min_minpct]
    if search.strip():
        pool = pool[pool["Player"].astype(str).str.contains(search.strip(), case=False, na=False)]

    # 不使用 UConn 的 fit score；若當前排序欄位不存在就退而用 ortg
    valid_sort = sort_by if sort_by in pool.columns else ("ortg" if "ortg" in pool.columns else None)
    if valid_sort:
        pool = pool.sort_values(valid_sort, ascending=False)

    # 只顯示 NCAA 通用欄位（移除 UConn 專屬）
    base_rank_cols = [
        "Player",
        "Team",
        ("role" if "role" in pool.columns else role_col),
        "games","min_pct","ortg",
        "efg","ts","2p_pct","3p_pct","ft_pct",
        "usg","ast%","to%",
        "ortg_pctl","efg_pctl","ts_pctl","2p_pct_pctl","3p_pct_pctl","ft_pct_pctl","usg_pctl","ast%_pctl","tov_inv_pctl"
    ]
    rank_cols = [c for c in base_rank_cols if isinstance(c, str) and c in pool.columns]


    st.subheader(f"{filtered_title} — {title_suffix}")
    st.dataframe(pool[rank_cols].head(top_k), use_container_width=True)
    st.download_button(
        "Download current table (CSV)",
        pool[rank_cols].head(top_k).to_csv(index=False).encode("utf-8"),
        file_name=dl_name,
        mime="text/csv"
    )

else:
    pool = u_join.copy()
    if role_col and role_pick != "All":
        pool = pool[pool[role_col].astype(str) == role_pick]
    if "min_pct" in pool.columns:
        pool = pool[pool["min_pct"].fillna(0) >= min_minpct]
    if search.strip():
        pool = pool[pool["Player"].astype(str).str.contains(search.strip(), case=False, na=False)]
    pool = add_fit_score(pool)
    if sort_by in pool.columns:
        pool = pool.sort_values(sort_by, ascending=False)

    st.subheader("Rankings — UConn Players Only")
    rank_cols = [c for c in [
        "Player", role_col, "Team","Prev_Team",
        "games","min_pct","ortg","proj_ortg_uconn","delta_ortg_uconn","fit_score_uconn_0_100",
        "efg","ts","2p_pct","3p_pct","ft_pct","usg","ast%","to%",
        "ortg_pctl","efg_pctl","ts_pctl","2p_pct_pctl","3p_pct_pctl","ft_pct_pctl","usg_pctl","ast%_pctl","tov_inv_pctl"
    ] if isinstance(c, str) and c in pool.columns]
    st.dataframe(pool[rank_cols].head(top_k), use_container_width=True)
    st.download_button(
        "Download current table (CSV)",
        pool[rank_cols].head(top_k).to_csv(index=False).encode("utf-8"),
        file_name="uconn_roster_rankings.csv",
        mime="text/csv"
    )

# ======== UConn-only Filters + Rankings (independent of NCAA section) ========
with st.expander("Filters (UConn only)", expanded=False):
    role_opts_uc = ["All"]
    if role_col:
        role_opts_uc += sorted(u_join[role_col].dropna().astype(str).unique())
    role_pick_uc = st.selectbox("Role (UConn)", role_opts_uc, index=0, key="uc_role_pick")

    min_minpct_uc = 10 if "min_pct" in u_join.columns else 0
    if "min_pct" in u_join.columns:
        min_minpct_uc = st.slider("Min% (minutes share) ≥ (UConn)", 0, 80, 10, step=5, key="uc_minpct")

    sort_candidates_uc = [
        c for c in [
            "proj_ortg_uconn","delta_ortg_uconn","fit_score_uconn_0_100","ortg",
            "efg","ts","3p_pct","2p_pct","ft_pct","usg"
        ] if c in u_join.columns
    ]
    sort_by_uc = st.selectbox("Sort by (UConn)", sort_candidates_uc, index=0 if sort_candidates_uc else 0, key="uc_sortby")

    top_k_uc = st.slider("Show Top K (UConn)", 5, 50, 15, step=5, key="uc_topk")
    search_uc = st.text_input("Search player name (UConn contains)", key="uc_search")

pool_uc = u_join.copy()
if role_col and role_pick_uc != "All":
    pool_uc = pool_uc[pool_uc[role_col].astype(str) == role_pick_uc]
if "min_pct" in pool_uc.columns:
    pool_uc = pool_uc[pool_uc["min_pct"].fillna(0) >= min_minpct_uc]
if search_uc.strip():
    pool_uc = pool_uc[pool_uc["Player"].astype(str).str.contains(search_uc.strip(), case=False, na=False)]

pool_uc = add_fit_score(pool_uc)
if sort_by_uc in pool_uc.columns:
    pool_uc = pool_uc.sort_values(sort_by_uc, ascending=False)

st.subheader("Rankings — UConn (local filters)")
rank_cols_uc = [c for c in [
    "Player", role_col, "Team","Prev_Team",
    "games","min_pct","ortg","proj_ortg_uconn","delta_ortg_uconn","fit_score_uconn_0_100",
    "efg","ts","2p_pct","3p_pct","ft_pct","usg","ast%","to%",
    "ortg_pctl","efg_pctl","ts_pctl","2p_pct_pctl","3p_pct_pctl","ft_pct_pctl","usg_pctl","ast%_pctl","tov_inv_pctl"
] if isinstance(c, str) and c in pool_uc.columns]

st.dataframe(pool_uc[rank_cols_uc].head(top_k_uc), use_container_width=True)
st.download_button(
    "Download UConn table (CSV)",
    pool_uc[rank_cols_uc].head(top_k_uc).to_csv(index=False).encode("utf-8"),
    file_name="rankings_uconn_local.csv",
    mime="text/csv"
)
# ======== end UConn-only Filters + Rankings ========

# ----------------------------- Rotation planner -----------------------------
st.subheader("Projected Rotation Minutes")
if "rot_minutes" not in st.session_state:
    est = []
    for _, r in u_join.iterrows():
        if "min_pct" in u_join.columns and not pd.isna(r.get("min_pct")):
            est.append(round(float(r["min_pct"])*0.40, 1))
        else:
            est.append(12.0)
    st.session_state.rot_minutes = dict(zip(u_join["Player"].astype(str), est))

cols = st.columns(4)
with cols[0]:
    if st.button("Autofill from last year's Min%"):
        new = {}
        for _, r in u_join.iterrows():
            new[r["Player"]] = round(float(r.get("min_pct", 0))*0.40, 1) if not pd.isna(r.get("min_pct")) else 12.0
        st.session_state.rot_minutes = new
with cols[1]:
    if st.button("Reset to 0"):
        st.session_state.rot_minutes = {p: 0.0 for p in u_join["Player"].astype(str)}
with cols[2]:
    if st.button("Scale to 200"):
        vals = np.array(list(st.session_state.rot_minutes.values()), dtype=float)
        s = vals.sum()
        if s > 0:
            vals = vals*200.0/s
            st.session_state.rot_minutes = dict(zip(st.session_state.rot_minutes.keys(), np.round(vals,1)))

mins_inputs = {}
grid = st.columns(5)
for i, p in enumerate(u_join["Player"].astype(str)):
    with grid[i%5]:
        mins_inputs[p] = st.number_input(p, min_value=0.0, max_value=40.0, step=1.0,
                                         value=float(st.session_state.rot_minutes.get(p,0.0)))
st.session_state.rot_minutes = mins_inputs
total_mins = round(sum(st.session_state.rot_minutes.values()),1)
st.caption(f"Total minutes = **{total_mins:.1f}**  (target = 200)")

# ----------------------------- Lineup Impact -----------------------------
st.subheader("Lineup Impact")
all_players = u_join["Player"].astype(str).tolist()
default_a = [p for p,_ in sorted(st.session_state.rot_minutes.items(), key=lambda x: x[1], reverse=True)[:5]]
lineup_a = st.multiselect("Lineup A (pick 5)", all_players, default=default_a, max_selections=5)
lineup_b = st.multiselect("Lineup B (optional, pick 5)", all_players, max_selections=5)

def safe_proj(players):
    if len(players)!=5: return np.nan
    return lineup_proj_ortg(u_join, players, three_pref, afgm_pref, tov_pref)

ortg_a = safe_proj(lineup_a)
ortg_b = safe_proj(lineup_b) if lineup_b else np.nan

c1,c2,c3 = st.columns(3)
with c1:
    st.metric("Lineup A — projected ORtg", f"{ortg_a:.1f}" if not pd.isna(ortg_a) else "—")
with c2:
    if lineup_b and len(lineup_b)==5 and not pd.isna(ortg_a) and not pd.isna(ortg_b):
        st.metric("Lineup B — projected ORtg", f"{ortg_b:.1f}")
with c3:
    if lineup_b and len(lineup_b)==5 and not pd.isna(ortg_a) and not pd.isna(ortg_b):
        st.metric("Impact (A − B)", f"{(ortg_a-ortg_b):+.1f}")

# ----------------------------- Player Detail -----------------------------
st.subheader("Player Detail")

ncaa_school_pick = st.session_state.get("ncaa_school_pick", "— None —")
ncaa_player_pick = st.session_state.get("ncaa_player_pick", None)

def _radar_from_row(row: pd.Series, cols_available: List[str]):
    axes = [
        ("efg_pctl","eFG%"), ("ts_pctl","TS%"),
        ("2p_pct_pctl","2P%"), ("3p_pct_pctl","3P%"),
        ("ft_pct_pctl","FT%"), ("usg_pctl","USG%"),
        ("ast%_pctl","AST%"), ("tov_inv_pctl","TOV (inv)"),
        ("dr%_pctl","DR%"), ("or%_pctl","OR%"),
        ("stl%_pctl","STL%"), ("blk%_pctl","BLK%"),
    ]
    cats, vals = [], []
    for key, label in axes:
        val = None
        k_best = key.replace("_pctl","_pctl_best")
        if k_best in cols_available and pd.notna(row.get(k_best)):
            val = float(row.get(k_best))
        elif key in cols_available and pd.notna(row.get(key)):
            val = float(row.get(key))
        else:
            raw_key = key.replace("_pctl","").replace("tov_inv","to%")
            if raw_key in cols_available and pd.notna(row.get(raw_key)):
                v = float(row.get(raw_key))
                if ("pct" in raw_key) or raw_key.endswith("%") or raw_key in ("efg","ts","usg","ast%","to%","dr%","or%","stl%","blk%"):
                    if v<=1: v = v*100
                    v = max(0.0, min(100.0, v))
                val = v
        if val is not None:
            cats.append(label); vals.append(val)
    return cats, vals

if (not ncaa.empty) and ncaa_school_pick and (ncaa_school_pick != "— None —") and ncaa_player_pick:
    ncaa_pool = ncaa[ncaa["Team"].astype(str) == ncaa_school_pick]
    row = ncaa_pool[ncaa_pool["Player"]==ncaa_player_pick]
    if not row.empty:
        prow = row.iloc[0]
        cats, vals = _radar_from_row(prow, ncaa.columns.tolist())
        if cats:
            fig = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself', name='Percentile'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to draw radar.")
    else:
        st.info("Selected NCAA player not found under the chosen school.")
    st.caption("Tip: Clear the NCAA School picker in Filters to switch back to UConn player details.")
else:
    pick = st.selectbox("Choose a UConn player", sorted(u_join["Player"].dropna().astype(str).unique()))
    prow = u_join[u_join["Player"]==pick].iloc[0]
    cats, vals = _radar_from_row(prow, u_join.columns.tolist())
    if cats:
        fig = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself', name='Percentile/Scaled'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to draw radar.")
