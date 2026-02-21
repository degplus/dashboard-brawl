# pages/1_Player_Scout.py
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import plotly.express as px
import json
import requests
import base64
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Player Scout — DegStats",
    page_icon="🔍"
    layout="wide"
)

# ============================================================
# BIGQUERY CLIENT
# ============================================================
@st.cache_resource
def get_bq_client():
    project_id = st.secrets.get("gcp_project", "brawl-sandbox")
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials, project=project_id)

client = get_bq_client()
with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("---")


# ============================================================
# FETCH DATA — BigQuery
# ============================================================
@st.cache_data(ttl=3600, persist="disk")
def fetch_data(query: str, params_json: str = None) -> pd.DataFrame:
    bq_params = []
    if params_json:
        for p in json.loads(params_json):
            param_cls = (
                bigquery.ArrayQueryParameter
                if isinstance(p["value"], list)
                else bigquery.ScalarQueryParameter
            )
            bq_params.append(param_cls(p["name"], p["bq_type"], p["value"]))
    job_config = bigquery.QueryJobConfig(query_parameters=bq_params)
    return client.query(query, job_config=job_config).result().to_dataframe()

# ============================================================
# IMAGE → BASE64
# ============================================================
@st.cache_data(ttl=86400, persist="disk")
def img_to_base64(url: str) -> str | None:
    if not url:
        return None
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if resp.status_code == 200:
            mime = resp.headers.get("Content-Type", "image/png").split(";")[0]
            b64 = base64.b64encode(resp.content).decode()
            return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def convert_img_column(series: pd.Series) -> pd.Series:
    urls = series.tolist()
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(img_to_base64, url): i for i, url in enumerate(urls)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
    return pd.Series([results[i] for i in range(len(urls))])

# ============================================================
# BRAWL STARS API — PLAYER PROFILE (via bsproxy)
# ============================================================
BRAWL_API_BASE = "https://bsproxy.royaleapi.dev/v1"

@st.cache_data(ttl=300)
def fetch_api_profile(tag: str) -> dict | None:
    token = st.secrets.get("BRAWL_API_TOKEN", "")
    if not token:
        return {"_error": "no_token"}
    encoded = tag.replace("#", "%23")
    url = f"{BRAWL_API_BASE}/players/{encoded}"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=8
        )
        if resp.status_code == 200:
            return resp.json()
        return {"_error": f"http_{resp.status_code}", "_detail": resp.text[:200]}
    except Exception as e:
        return {"_error": "exception", "_detail": str(e)}

# ============================================================
# LOAD ALL PLAYERS (active + scouting targets)
# ============================================================
@st.cache_data(ttl=3600, persist="disk")
def load_all_players() -> pd.DataFrame:
    return fetch_data("""
        SELECT
            f.player_tag,
            ARRAY_AGG(f.player_name ORDER BY f.battle_date DESC LIMIT 1)[OFFSET(0)] AS display_name,
            ARRAY_AGG(f.player_team ORDER BY f.battle_date DESC LIMIT 1)[OFFSET(0)] AS last_team,
            MAX(f.battle_date) AS last_seen,
            COALESCE(MAX(CASE WHEN sp.is_active = TRUE THEN 1 ELSE 0 END), 0) = 1 AS is_active
        FROM `brawl-sandbox.brawl_stats.dim_filters` f
        LEFT JOIN `brawl-sandbox.brawl_stats.dim_source_players` sp
            ON f.player_tag = sp.PL_TAG AND sp.is_active = TRUE
        GROUP BY f.player_tag
        ORDER BY last_seen DESC
    """)

# ============================================================
# PLAYER DATE RANGE
# ============================================================
@st.cache_data(ttl=3600, persist="disk")
def get_player_date_range(tag: str):
    df = fetch_data("""
        SELECT
            MIN(battle_date) AS min_date,
            MAX(battle_date) AS max_date
        FROM `brawl-sandbox.brawl_stats.dim_filters`
        WHERE player_tag = @tag
    """, json.dumps([{"name": "tag", "bq_type": "STRING", "value": tag}]))
    if df.empty:
        return None, None
    return df["min_date"].iloc[0], df["max_date"].iloc[0]

# ============================================================
# DRAFT VIEWER HELPER
# ============================================================
def render_draft(game_id: int, map_name: str, map_img: str, bt: str):
    df_draft = fetch_data("""
        SELECT team_num, player_name, player_tag, player_team, player_result,
               brawler_name, brawler_img, star_player_name
        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        WHERE game = @game_id
        ORDER BY team_num, player_place
    """, json.dumps([{"name": "game_id", "bq_type": "INT64", "value": game_id}]))

    if df_draft.empty:
        st.warning("Draft data not found.")
        return

    blue = df_draft[df_draft["team_num"] == 1].reset_index(drop=True)
    red  = df_draft[df_draft["team_num"] == 2].reset_index(drop=True)

    star_player    = df_draft["star_player_name"].iloc[0]
    blue_result    = blue["player_result"].iloc[0] if not blue.empty else ""
    red_result     = red["player_result"].iloc[0]  if not red.empty  else ""
    blue_team_name = blue["player_team"].iloc[0]   if not blue.empty else "Blue"
    red_team_name  = red["player_team"].iloc[0]    if not red.empty  else "Red"
    blue_emoji     = "🏆" if blue_result == "victory" else "💀"
    red_emoji      = "🏆" if red_result  == "victory" else "💀"

    col_blue, col_center, col_red = st.columns([4, 2, 4])

    with col_blue:
        st.markdown(
            f"<h3 style='color:#4A90D9;text-align:center'>{blue_team_name}</h3>",
            unsafe_allow_html=True
        )
        h1, h2, h3 = st.columns(3)
        h1.markdown("**Team**")
        h2.markdown("**Player**")
        h3.markdown("**Pick**")
        for _, row in blue.iterrows():
            c1, c2, c3 = st.columns(3)
            c1.write(row["player_team"] or "-")
            c2.write(row["player_name"])
            if row["brawler_img"]:
                c3.image(row["brawler_img"], width=48)
            c3.caption(row["brawler_name"])
        st.markdown(
            f"<h4 style='text-align:center'>{blue_emoji} {'VICTORY' if blue_result == 'victory' else 'DEFEAT'}</h4>",
            unsafe_allow_html=True
        )

    with col_center:
        st.markdown(f"<h4 style='text-align:center'>{map_name}</h4>", unsafe_allow_html=True)
        if map_img:
            st.image(map_img, use_container_width=True)
        st.markdown(f"<p style='text-align:center'>⭐ <b>{star_player}</b></p>", unsafe_allow_html=True)
        st.caption(bt)

    with col_red:
        st.markdown(
            f"<h3 style='color:#D94A4A;text-align:center'>{red_team_name}</h3>",
            unsafe_allow_html=True
        )
        h1, h2, h3 = st.columns(3)
        h1.markdown("**Pick**")
        h2.markdown("**Player**")
        h3.markdown("**Team**")
        for _, row in red.iterrows():
            c1, c2, c3 = st.columns(3)
            if row["brawler_img"]:
                c1.image(row["brawler_img"], width=48)
            c1.caption(row["brawler_name"])
            c2.write(row["player_name"])
            c3.write(row["player_team"] or "-")
        st.markdown(
            f"<h4 style='text-align:center'>{red_emoji} {'VICTORY' if red_result == 'victory' else 'DEFEAT'}</h4>",
            unsafe_allow_html=True
        )

# ============================================================
# HEADER
# ============================================================
st.title("🔍 Player Scout")
st.caption("Analyse any player independently — active roster or scouting target.")
st.markdown("---")

# ============================================================
# LOAD PLAYERS
# ============================================================
df_all_players = load_all_players()

if df_all_players.empty:
    st.error("No player data found.")
    st.stop()

df_all_players["last_seen"] = pd.to_datetime(df_all_players["last_seen"]).dt.date

name_to_tag = dict(zip(df_all_players["display_name"], df_all_players["player_tag"]))
tag_to_row  = df_all_players.set_index("player_tag")

# ============================================================
# CONTROLS: PLAYER + DATE
# ============================================================
ctrl1, ctrl2 = st.columns([2, 2])

with ctrl1:
    player_options = sorted(df_all_players["display_name"].tolist())
    selected_name  = st.selectbox("👤 Select Player", options=player_options)

selected_tag = name_to_tag.get(selected_name)
if not selected_tag:
    st.error("Player tag not found.")
    st.stop()

p_min_raw, p_max_raw = get_player_date_range(selected_tag)
if p_min_raw is None:
    st.warning("No data found for this player.")
    st.stop()

p_min = pd.to_datetime(p_min_raw).date()
p_max = pd.to_datetime(p_max_raw).date()

with ctrl2:
    scout_dates = st.date_input(
        "📅 Period",
        value=(p_min, p_max),
        min_value=p_min,
        max_value=p_max,
        key="scout_dates"
    )

if isinstance(scout_dates, (tuple, list)) and len(scout_dates) == 2:
    scout_start, scout_end = scout_dates
else:
    scout_start = scout_end = (scout_dates[0] if isinstance(scout_dates, (tuple, list)) else scout_dates)

# ============================================================
# PLAYER HEADER CARD (BigQuery)
# ============================================================
player_row  = tag_to_row.loc[selected_tag]
is_active   = bool(player_row["is_active"])
last_team   = player_row["last_team"] or "—"
last_seen   = str(player_row["last_seen"])[:10]
badge       = "🟢 Active Roster" if is_active else "🔵 Scouting Target"
badge_color = "#1a7f37" if is_active else "#0969da"

p_img_df = fetch_data("""
    SELECT player_img
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
    WHERE player_tag = @tag
    ORDER BY battle_time DESC
    LIMIT 1
""", json.dumps([{"name": "tag", "bq_type": "STRING", "value": selected_tag}]))

card_img, card_info = st.columns([1, 7])
with card_img:
    if not p_img_df.empty and p_img_df["player_img"].iloc[0]:
        b64 = img_to_base64(p_img_df["player_img"].iloc[0])
        if b64:
            st.image(b64, width=80)
with card_info:
    st.markdown(
        f"### {selected_name} &nbsp;"
        f"<span style='background:{badge_color};color:white;"
        f"padding:3px 10px;border-radius:12px;font-size:0.8rem'>{badge}</span>",
        unsafe_allow_html=True
    )
    st.caption(f"**Team:** {last_team}  |  **Tag:** `{selected_tag}`  |  **Last seen:** {last_seen}")

st.markdown("---")

# ============================================================
# LIVE PROFILE — Brawl Stars API
# ============================================================
profile = fetch_api_profile(selected_tag)

if profile and "_error" in profile:
    err = profile["_error"]
    detail = profile.get("_detail", "")
    if err == "no_token":
        st.warning("⚠️ BRAWL_API_TOKEN não encontrado nos secrets.")
    elif err.startswith("http_"):
        st.warning(f"⚠️ API retornou status {err.replace('http_', '')} — {detail}")
    else:
        st.warning(f"⚠️ Erro ao chamar a API: {detail}")
    profile = None

# Monta dict de win streak por brawler (para cruzar com BigQuery)
api_streak_map = {}
if profile:
    for b in profile.get("brawlers", []):
        api_streak_map[b.get("name", "").upper()] = b.get("maxWinStreak", 0)

if profile:
    st.subheader("🎖️ Live Profile")

    club     = profile.get("club") or {}
    icon_id  = (profile.get("icon") or {}).get("id")
    icon_url = f"https://cdn.brawlify.com/profile-icons/regular/{icon_id}.png" if icon_id else None
    is_champion = profile.get("isQualifiedFromChampionshipChallenge", False)

    ic1, ic2 = st.columns([1, 11])
    with ic1:
        if icon_url:
            icon_b64 = img_to_base64(icon_url)
            if icon_b64:
                st.image(icon_b64, width=56)
    with ic2:
        name_color_raw = profile.get("nameColor", "0xffffffff")
        name_color     = "#" + name_color_raw.replace("0xff", "").replace("0x", "")
        champion_badge = (
            "&nbsp;<span style='background:#f0a500;color:#000;"
            "padding:2px 8px;border-radius:10px;font-size:0.75rem'>"
            "🏅 Championship Qualified</span>"
            if is_champion else ""
        )
        st.markdown(
            f"<span style='color:{name_color};font-size:1.3rem;font-weight:bold;"
            f"text-shadow:1px 1px 3px #000,-1px -1px 3px #000,"
            f"1px -1px 3px #000,-1px 1px 3px #000'>"
            f"{profile.get('name', selected_name)}</span>{champion_badge}",
            unsafe_allow_html=True
        )
        if club:
            st.caption(f"🤝 Club: **{club.get('name', '—')}** `{club.get('tag', '')}`  |  "
                       f"Exp Lvl {profile.get('expLevel', '—')} ({profile.get('expPoints', 0):,} XP)")

    # KPIs da API
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("🏆 Trophies",         f"{profile.get('trophies', 0):,}")
    a2.metric("📈 Highest Trophies", f"{profile.get('highestTrophies', 0):,}")
    a3.metric("🎮 3v3 Victories",    f"{profile.get('3vs3Victories', 0):,}")
    a4.metric("🥊 Solo Victories",   f"{profile.get('soloVictories', 0):,}")
    a5.metric("👥 Duo Victories",    f"{profile.get('duoVictories', 0):,}")

    # Brawler Roster da API
    brawlers_api = profile.get("brawlers", [])
    if brawlers_api:
        st.markdown("---")
        st.subheader(f"🔓 Brawler Roster — {len(brawlers_api)} unlocked")

        df_api_brawlers = pd.DataFrame([
            {
                "brawler_name":     b.get("name", ""),
                "power":            b.get("power", 0),
                "rank":             b.get("rank", 0),
                "trophies":         b.get("trophies", 0),
                "highest_trophies": b.get("highestTrophies", 0),
                "max_win_streak":   b.get("maxWinStreak", 0),
                "gadgets":          len(b.get("gadgets", [])),
                "star_powers":      len(b.get("starPowers", [])),
                "gears":            len(b.get("gears", [])),
            }
            for b in brawlers_api
        ]).sort_values("trophies", ascending=False).reset_index(drop=True)

        max_trophies = int(df_api_brawlers["trophies"].max()) + 1 if not df_api_brawlers.empty else 1000

        # Roster Summary
        df_full = df_api_brawlers.copy()
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("⚡ Max Power (11)",     int((df_full["power"] == 11).sum()))
        s2.metric("🏅 Rank 35+",           int((df_full["rank"] >= 35).sum()))
        s3.metric("🔧 All Gadgets (2/2)",  int((df_full["gadgets"] == 2).sum()))
        s4.metric("⭐ All Star Powers",    int((df_full["star_powers"] == 2).sum()))
        s5.metric("⚙️ All Gears (2/2)",   int((df_full["gears"] == 2).sum()))

        st.dataframe(
            df_api_brawlers,
            use_container_width=True,
            column_config={
                "brawler_name":     st.column_config.TextColumn("Brawler"),
                "power":            st.column_config.NumberColumn("Power",        format="%d"),
                "rank":             st.column_config.NumberColumn("Rank",         format="%d"),
                "trophies":         st.column_config.ProgressColumn(
                                        "Trophies", format="%d",
                                        min_value=0, max_value=max_trophies
                                    ),
                "highest_trophies": st.column_config.NumberColumn("Highest 🏆",   format="%d"),
                "max_win_streak":   st.column_config.NumberColumn("🔥 Win Streak", format="%d"),
                "gadgets":          st.column_config.NumberColumn("Gadgets",      format="%d"),
                "star_powers":      st.column_config.NumberColumn("Star Powers",  format="%d"),
                "gears":            st.column_config.NumberColumn("Gears",        format="%d"),
            },
            hide_index=True
        )

else:
    st.caption("⚠️ Live profile unavailable — check API token or player accessibility.")

st.markdown("---")

# ============================================================
# SCOUT WHERE + PARAMS (BigQuery)
# ============================================================
scout_where  = "player_tag = @tag AND DATE(battle_time) BETWEEN @d_start AND @d_end"
scout_params = json.dumps([
    {"name": "tag",     "bq_type": "STRING", "value": selected_tag},
    {"name": "d_start", "bq_type": "DATE",   "value": str(scout_start)},
    {"name": "d_end",   "bq_type": "DATE",   "value": str(scout_end)},
])

# ============================================================
# KPIs — BigQuery
# ============================================================
st.subheader("📊 Stats in Period")

df_kpi = fetch_data(f"""
    SELECT
        COUNT(DISTINCT game) AS games,
        COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
        COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
        SAFE_DIVIDE(
            COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) * 100.0,
            COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END)
        ) AS win_rate
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
    WHERE {scout_where}
""", scout_params)

if not df_kpi.empty:
    k = df_kpi.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎮 Games",    int(k["games"]))
    c2.metric("✅ Wins",     int(k["wins"]))
    c3.metric("❌ Losses",   int(k["losses"]))
    c4.metric("📊 Win Rate", f"{k['win_rate']:.1f}%" if k["win_rate"] is not None else "—")

st.markdown("---")

# ============================================================
# BRAWLER POOL — BigQuery + win streak da API
# ============================================================
st.header("👊 Brawler Pool")

df_pool = fetch_data(f"""
    WITH total AS (
        SELECT COUNT(DISTINCT game) AS total_games
        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        WHERE {scout_where}
    )
    SELECT
        MAX(brawler_img)  AS brawler_img,
        brawler_name,
        COUNT(*)          AS picks,
        SAFE_DIVIDE(COUNT(*) * 100.0, MAX(total.total_games)) AS pick_rate,
        COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
        COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
        SAFE_DIVIDE(
            COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) * 100.0,
            COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END)
        ) AS win_rate
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`, total
    WHERE {scout_where}
    GROUP BY brawler_name
    ORDER BY picks DESC
""", scout_params)

if df_pool.empty:
    st.warning("No brawler data for the selected period.")
else:
    # Cruza com win streak da API
    if api_streak_map:
        df_pool["max_win_streak"] = df_pool["brawler_name"].str.upper().map(api_streak_map).fillna(0).astype(int)
    else:
        df_pool["max_win_streak"] = 0

    pool_col_order = ["brawler_img", "brawler_name", "picks", "pick_rate", "wins", "losses", "win_rate", "max_win_streak"]
    pool_col_config = {
        "brawler_img":    st.column_config.ImageColumn(""),
        "brawler_name":   st.column_config.TextColumn("Brawler"),
        "picks":          st.column_config.NumberColumn("Picks",        format="%d"),
        "pick_rate":      st.column_config.ProgressColumn("Pick Rate",   format="%.1f%%", min_value=0, max_value=100),
        "wins":           st.column_config.NumberColumn("Wins",          format="%d"),
        "losses":         st.column_config.NumberColumn("Losses",        format="%d"),
        "win_rate":       st.column_config.ProgressColumn("Win Rate",    format="%.1f%%", min_value=0, max_value=100),
        "max_win_streak": st.column_config.NumberColumn("🔥 Win Streak", format="%d"),
    }

    st.dataframe(
        df_pool,
        use_container_width=True,
        column_order=pool_col_order,
        column_config=pool_col_config,
        hide_index=True
    )

st.markdown("---")

# ============================================================
# MAP PERFORMANCE — BigQuery
# ============================================================
st.header("🗺️ Map Performance")

df_maps = fetch_data(f"""
    SELECT
        MAX(map_img) AS map_img,
        map,
        mode,
        COUNT(DISTINCT game) AS games,
        COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
        COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
        SAFE_DIVIDE(
            COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) * 100.0,
            COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END)
        ) AS win_rate
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
    WHERE {scout_where}
    GROUP BY map, mode
    ORDER BY games DESC
""", scout_params)

if df_maps.empty:
    st.warning("No map data for the selected period.")
else:
    st.dataframe(
        df_maps,
        use_container_width=True,
        column_order=["map_img", "map", "mode", "games", "wins", "losses", "win_rate"],
        column_config={
            "map_img":  st.column_config.ImageColumn("Preview"),
            "map":      st.column_config.TextColumn("Map"),
            "mode":     st.column_config.TextColumn("Mode"),
            "games":    st.column_config.NumberColumn("Games",    format="%d"),
            "wins":     st.column_config.NumberColumn("Wins",     format="%d"),
            "losses":   st.column_config.NumberColumn("Losses",   format="%d"),
            "win_rate": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
        },
        hide_index=True
    )

st.markdown("---")

# ============================================================
# TIMELINE — BigQuery
# ============================================================
st.header("📈 Timeline")

t1, t2 = st.columns(2)
with t1:
    granularity = st.radio("Granularity", ["Day", "Week"], horizontal=True, key="scout_gran")
with t2:
    wr_type = st.radio("Win Rate", ["Daily", "Cumulative"], horizontal=True, key="scout_wr")

period_expr = "DATE(battle_time)" if granularity == "Day" else "DATE_TRUNC(DATE(battle_time), WEEK)"

df_tl = fetch_data(f"""
    SELECT
        {period_expr} AS period,
        COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
        COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
        COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END) AS total_games
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
    WHERE {scout_where}
    GROUP BY period
    ORDER BY period
""", scout_params)

if df_tl.empty:
    st.warning("No timeline data for the selected period.")
else:
    df_tl["period"] = pd.to_datetime(df_tl["period"])
    df_tl = df_tl.sort_values("period")

    if wr_type == "Daily":
        df_tl["win_rate"] = (df_tl["wins"] / df_tl["total_games"] * 100).round(1)
        ylabel = "Win Rate — Daily"
    else:
        df_tl["cum_wins"]  = df_tl["wins"].cumsum()
        df_tl["cum_games"] = df_tl["total_games"].cumsum()
        df_tl["win_rate"]  = (df_tl["cum_wins"] / df_tl["cum_games"] * 100).round(1)
        ylabel = "Win Rate — Cumulative"

    df_tl["games_label"] = df_tl["total_games"].astype(str) + " games"

    fig = px.line(
        df_tl,
        x="period",
        y="win_rate",
        markers=True,
        hover_data={"games_label": True, "total_games": False},
        labels={"period": "Date", "win_rate": ylabel, "games_label": "Games"},
        title=f"{selected_name} — Win Rate over Time ({granularity} / {wr_type})"
    )
    fig.add_hline(
        y=50, line_dash="dash", line_color="gray", opacity=0.5,
        annotation_text="50%", annotation_position="bottom right"
    )
    fig.update_layout(
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        xaxis_title="Date",
        yaxis_title=ylabel,
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================
# RECENT GAMES — BigQuery
# ============================================================
st.header("📋 Recent Games")

df_games = fetch_data(f"""
    SELECT DISTINCT
        game,
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', battle_time) AS battle_time,
        map,
        mode,
        player_result,
        player_team,
        MAX(map_img) AS map_img
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
    WHERE {scout_where}
    GROUP BY game, battle_time, map, mode, player_result, player_team
    ORDER BY battle_time DESC
    LIMIT 50
""", scout_params)

if df_games.empty:
    st.warning("No games found for the selected period.")
else:
    df_games["result_show"] = df_games["player_result"].apply(
        lambda x: "WIN" if x == "victory" else "LOSS"
    )

    st.caption("Click a row to view the full draft.")
    ev = st.dataframe(
        df_games,
        use_container_width=True,
        column_order=["game", "battle_time", "map", "mode", "result_show", "player_team"],
        column_config={
            "game":        st.column_config.NumberColumn("Game",      format="%d"),
            "battle_time": st.column_config.TextColumn("Date / Time"),
            "map":         st.column_config.TextColumn("Map"),
            "mode":        st.column_config.TextColumn("Mode"),
            "result_show": st.column_config.TextColumn("Result"),
            "player_team": st.column_config.TextColumn("Team"),
        },
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    selected = ev.selection.rows
    if selected:
        row = df_games.iloc[selected[0]]
        st.markdown("---")
        render_draft(int(row["game"]), row["map"], row["map_img"], row["battle_time"])
