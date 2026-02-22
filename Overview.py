import streamlit as st
import streamlit_authenticator as stauth
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import plotly.express as px
import time
import datetime
from datetime import timezone
import json
import requests
import base64
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed



# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DegStats - Overview",
    page_icon="📊",
    layout="wide"
)

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

# ============================================================
# 🚫 LIMPEZA VISUAL (TENTATIVA MÁXIMA)
# ============================================================
st.markdown("""
    <style>
        /* 1. Esconde Cabeçalho, Menu Hamburger e Toolbar */
        header {visibility: hidden !important;}
        [data-testid="stHeader"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        
        /* 2. Esconde Rodapé Padrão */
        footer {visibility: hidden !important; height: 0px !important;}
        
        /* 3. Tenta esconder o Badge do Perfil (Bonequinho) */
        /* Procura por qualquer DIV que tenha 'viewerBadge' no nome da classe */
        div[class*="viewerBadge"] {
            visibility: hidden !important;
            display: none !important;
            opacity: 0 !important;
        }

        /* 4. Tenta esconder o botão 'Hosted with Streamlit' vermelho */
        .stApp > footer {
            display: none !important;
        }
        
        /* 5. Ajuste de margem para o conteúdo subir e cobrir o vazio */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }
    </style>
""", unsafe_allow_html=True)


def set_gradient_background():
    page_bg_img = """
    <style>
    .stApp {
        /* "to bottom" faz descer reto (vertical) */
        /* Cores: Começa bem escuro (#0e1117) e vai clareando para um azul noturno (#2a2d4a) */
        background-image: linear-gradient(to bottom, #0e1117, #1c1f33);
        background-attachment: fixed;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_gradient_background()

# Adiciona a logo no topo da sidebar
st.logo("assets/logo.png", icon_image="assets/logo.png")

from concurrent.futures import ThreadPoolExecutor, as_completed



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

# ============================================================
# CACHE LOAD TIME
# ============================================================
@st.cache_data(ttl=3600, persist="disk")
def get_cache_load_time():
    return time.time()

# ============================================================
# FETCH DATA
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
# DIM FILTERS
# ============================================================
@st.cache_data(ttl=3600, persist="disk")
def load_dim_filters() -> pd.DataFrame:
    df = fetch_data("SELECT * FROM `brawl-sandbox.brawl_stats.dim_filters`")
    df["battle_date"] = pd.to_datetime(df["battle_date"]).dt.date
    return df

# ============================================================
# PLAYER NAMES (active only)
# ============================================================
@st.cache_data(ttl=3600, persist="disk")
def load_player_names() -> dict:
    df = fetch_data("""
        SELECT v.player_tag, v.player_name AS display_name
        FROM (
            SELECT player_tag, player_name,
                ROW_NUMBER() OVER (
                    PARTITION BY player_tag ORDER BY battle_time DESC
                ) AS rn
            FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        ) v
        INNER JOIN `brawl-sandbox.brawl_stats.dim_source_players` sp
            ON v.player_tag = sp.PL_TAG AND sp.is_active = TRUE
        WHERE v.rn = 1
    """)
    return dict(zip(df["player_tag"], df["display_name"]))


# ============================================================
# ALL PLAYER NAMES
# ============================================================
@st.cache_data(ttl=3600, persist="disk")
def load_all_player_names() -> dict:
    df = fetch_data("""
        SELECT player_tag,
            ARRAY_AGG(player_name ORDER BY battle_time DESC LIMIT 1)[OFFSET(0)] AS display_name
        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        GROUP BY player_tag
    """)
    return dict(zip(df["player_tag"], df["display_name"]))


# ============================================================
# SESSION STATE
# ============================================================
FILTER_KEYS = ["f_region", "f_type", "f_mode", "f_map", "f_team", "f_player", "f_brawler"]
COL_MAP = {
    "f_region":  "source_player_region",
    "f_type":    "type",
    "f_mode":    "mode",
    "f_map":     "map",
    "f_team":    "player_team",
    "f_player":  "player_tag",
    "f_brawler": "brawler_name"
}

for key in FILTER_KEYS:
    if key not in st.session_state:
        st.session_state[key] = []

if st.session_state.get("clear_filters", False):
    for key in FILTER_KEYS:
        st.session_state[key] = []
    st.session_state["clear_filters"] = False

# ============================================================
# AUTO-REFRESH
# ============================================================
if time.time() - get_cache_load_time() > 3600:
    st.cache_data.clear()
    st.rerun()

# ============================================================
# LOAD BASE DATA
# ============================================================
df_dim           = load_dim_filters()
player_names     = load_player_names()
all_player_names = load_all_player_names()

# ============================================================
# HEADER
# ============================================================
st.title("📊 Overview")
st.caption("Brawl Stars — Competitive Match Analysis")
st.markdown("---")

# ============================================================
# SIDEBAR — DATE + TOGGLE
# ============================================================
with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("---")
    st.header("🔍 Filters")

    min_date = df_dim["battle_date"].min()
    max_date = df_dim["battle_date"].max()

    dates = st.date_input(
        "📅 Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        date_start, date_end = dates
    else:
        date_start = date_end = (dates[0] if isinstance(dates, (tuple, list)) else dates)

    st.markdown("---")

    show_only_active = st.toggle(
        "👤 Show only active roster",
        value=True,
        help=(
            "On: Players list shows only active roster members (is_active = TRUE). "
            "Off: Players list shows all players found in recorded matches."
        )
    )

    st.markdown("---")

# ============================================================
# GET FILTER OPTIONS
# ============================================================
def get_filter_options(exclude_key: str) -> list:
    mask = (
        (df_dim["battle_date"] >= date_start) &
        (df_dim["battle_date"] <= date_end)
    )

    for key, col in COL_MAP.items():
        if key == exclude_key:
            continue
        values = st.session_state.get(key, [])
        if not values:
            continue
        if key == "f_player":
            names_dict  = player_names if show_only_active else all_player_names
            name_to_tag = {v: k for k, v in names_dict.items()}
            tags = [name_to_tag.get(n) for n in values]
            tags = [t for t in tags if t]
            if not tags:
                mask &= False
            else:
                mask &= df_dim[col].isin(tags)
        else:
            mask &= df_dim[col].isin(values)

    col = COL_MAP[exclude_key]
    raw = sorted(df_dim[mask][col].dropna().unique().tolist())

    if exclude_key == "f_player":
        names_dict = player_names if show_only_active else all_player_names
        return sorted([names_dict[tag] for tag in raw if tag in names_dict])

    return raw

# ============================================================
# BUILD WHERE (COM SUPORTE A H2H)
# ============================================================
def build_where(use_datetime: bool = False):
    date_col = "DATE(battle_time)" if use_datetime else "battle_date"
    conds    = [f"{date_col} BETWEEN @d_start AND @d_end"]
    raw_params = [
        {"name": "d_start", "bq_type": "DATE", "value": str(date_start)},
        {"name": "d_end",   "bq_type": "DATE", "value": str(date_end)},
    ]

    # --- NOVO: FILTRO SQL PARA H2H ---
    # Se o modo H2H estiver ativo (e tiver 2 times), forçamos o filtro de jogo
    if is_h2h_mode and len(st.session_state.f_team) == 2:
        team_a = st.session_state.f_team[0]
        team_b = st.session_state.f_team[1]
        
        # Essa subquery mágica agrupa por jogo e garante que o count de times seja 2
        # filtrando apenas os jogos que contém AMBOS os times selecionados.
        conds.append(f"""
            game IN (
                SELECT game
                FROM `brawl-sandbox.brawl_stats.vw_battles_python`
                WHERE player_team IN (@h2h_t1, @h2h_t2)
                GROUP BY game
                HAVING COUNT(DISTINCT player_team) = 2
            )
        """)
        raw_params.append({"name": "h2h_t1", "bq_type": "STRING", "value": team_a})
        raw_params.append({"name": "h2h_t2", "bq_type": "STRING", "value": team_b})
    # ---------------------------------

    for key, col in COL_MAP.items():
        values = st.session_state.get(key, [])
        if not values:
            continue
        
        # Se estiver em H2H Mode, a gente PULA o filtro padrão de time
        # porque já fizemos o filtro especial ali em cima.
        if is_h2h_mode and key == "f_team":
            continue 

        if key == "f_player":
            names_dict  = player_names if show_only_active else all_player_names
            name_to_tag = {v: k for k, v in names_dict.items()}
            tags = [name_to_tag.get(n) for n in values]
            tags = [t for t in tags if t]
            if tags:
                conds.append(f"{col} IN UNNEST(@{key})")
                raw_params.append({"name": key, "bq_type": "STRING", "value": tags})
        else:
            conds.append(f"{col} IN UNNEST(@{key})")
            raw_params.append({"name": key, "bq_type": "STRING", "value": values})

    return " AND ".join(conds), json.dumps(raw_params)

# ============================================================
# BUILD WHERE H2H
# ============================================================
def build_where_h2h():
    conds      = ["DATE(battle_time) BETWEEN @d_start AND @d_end"]
    raw_params = [
        {"name": "d_start", "bq_type": "DATE", "value": str(date_start)},
        {"name": "d_end",   "bq_type": "DATE", "value": str(date_end)},
    ]

    for key, col in COL_MAP.items():
        if key == "f_team":
            continue
        values = st.session_state.get(key, [])
        if not values:
            continue
        if key == "f_player":
            names_dict  = player_names if show_only_active else all_player_names
            name_to_tag = {v: k for k, v in names_dict.items()}
            tags = [name_to_tag.get(n) for n in values]
            tags = [t for t in tags if t]
            if tags:
                conds.append(f"{col} IN UNNEST(@{key})")
                raw_params.append({"name": key, "bq_type": "STRING", "value": tags})
        else:
            conds.append(f"{col} IN UNNEST(@{key})")
            raw_params.append({"name": key, "bq_type": "STRING", "value": values})

    return " AND ".join(conds), raw_params

# ============================================================
# SIDEBAR — FILTERS
# ============================================================
filters_config = [
    ("f_region",  "Region",  "🌍"),
    ("f_type",    "Type",    "📌"),
    ("f_mode",    "Mode",    "🎮"),
    ("f_map",     "Map",     "🗺️"),
    ("f_team",    "Team",    "🏆"),
    ("f_player",  "Player",  "👤"),
    ("f_brawler", "Brawler", "👊"),
]

is_h2h_mode = False  # Variável para controlar o modo

with st.sidebar:
    for i, (key, label, emoji) in enumerate(filters_config):
        options = get_filter_options(key)
        st.multiselect(
            f"{emoji} {label}",
            options=options,
            key=key,
            placeholder=f"All {label.lower()}s..."
        )
        
        # --- NOVO: LÓGICA DO H2H ---
        if key == "f_team":
            # Se tiver exatamente 2 times selecionados, oferece o filtro estrito
            if len(st.session_state.f_team) == 2:
                is_h2h_mode = st.toggle(
                    "⚔️ H2H Mode (Only Direct Matches)",
                    value=False,
                    key="h2h_toggle_active",  # <--- ADICIONAMOS ESSA KEY IMPORTANTE
                    help="If active, shows ONLY metrics from games where these two teams played against each other."
                )
            else:
                # Garante que se mudar os times, o toggle reseta ou desliga
                is_h2h_mode = False
        # ---------------------------

        if i == 0:
            st.markdown("---")

    st.markdown("---")
    if any(st.session_state[k] for k in FILTER_KEYS):
        if st.button("🗑️ Clear All Filters", use_container_width=True):
            st.session_state["clear_filters"] = True
            st.rerun()

    cache_ts  = get_cache_load_time()
    loaded_at = datetime.datetime.fromtimestamp(cache_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mins_ago  = int((time.time() - cache_ts) / 60)
    ago_text  = "just now" if mins_ago == 0 else "1 min ago" if mins_ago == 1 else f"{mins_ago} min ago"

    st.success(f"⚡ Cache loaded at {loaded_at}")
    st.caption(f"🕐 Updated {ago_text}")

   
# ============================================================
# WHERE CLAUSE FINAL
# ============================================================
where_main, params_main = build_where(use_datetime=True)

# ============================================================
# DRAFT VIEWER HELPER
# ============================================================
def render_draft(game_id: int, map_name: str, map_img: str, bt: str):
    df_draft = fetch_data("""
        SELECT team_num, player_name, player_team, player_result,
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
        st.caption(f"{bt}")

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
# KPIs
# ============================================================
df_kpis = fetch_data(f"""
    SELECT
        COUNT(*)                   AS total_records,
        COUNT(DISTINCT game)       AS total_games,
        COUNT(DISTINCT player_tag) AS unique_players,
        MAX(battle_time)           AS last_update
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
    WHERE {where_main}
""", params_main)

if not df_kpis.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  f"{df_kpis['total_records'][0]:,}")
    c2.metric("Total Games",    f"{df_kpis['total_games'][0]:,}")
    c3.metric("Unique Players", f"{df_kpis['unique_players'][0]:,}")
    
    # Customização para diminuir a fonte do Last Update
    last_update_str = str(df_kpis["last_update"][0])[:16] + " UTC"
    c4.markdown(f"""
        <div style="font-size: 14px; color: #808495; margin-bottom: 4px;">Last Update</div>
        <div style="font-size: 24px; font-weight: 600;">{last_update_str}</div>
    """, unsafe_allow_html=True)
# ============================================================
# MAPS
# ============================================================
st.header("🗺️ Maps")

df_maps = fetch_data(f"""
    SELECT
        MAX(map_img) AS map_img,
        map,
        mode,
        COUNT(DISTINCT game) AS games,
        SAFE_DIVIDE(COUNT(DISTINCT game) * 100.0,
            SUM(COUNT(DISTINCT game)) OVER ()) AS pct_total,
        COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
        COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
        SAFE_DIVIDE(
            COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) * 100.0,
            COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END)
        ) AS win_rate
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
    WHERE {where_main}
    GROUP BY map, mode
    ORDER BY games DESC
""", params_main)

if df_maps.empty:
    st.warning("No data found for the selected filters.")
else:
    st.dataframe(
        df_maps,
        use_container_width=True,
        column_order=["map_img", "map", "mode", "games", "pct_total", "wins", "losses", "win_rate"],
        column_config={
            "map_img":   st.column_config.ImageColumn("Preview"),
            "map":       st.column_config.TextColumn("Map"),
            "mode":      st.column_config.TextColumn("Mode"),
            "games":     st.column_config.NumberColumn("Games",    format="%d"),
            "pct_total": st.column_config.ProgressColumn("Game Rate", format="%.1f%%", min_value=0, max_value=100),
            "wins":      st.column_config.NumberColumn("Wins",     format="%d"),
            "losses":    st.column_config.NumberColumn("Losses",   format="%d"),
            "win_rate":  st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
        },
        hide_index=True
    )

st.markdown("---")

# ============================================================
# BRAWLERS
# ============================================================
st.header("👊 Brawlers")

df_brawlers = fetch_data(f"""
    WITH total AS (
        SELECT COUNT(DISTINCT game) AS total_games
        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        WHERE {where_main}
    )
    SELECT
        MAX(brawler_img) AS brawler_img,
        brawler_name,
        COUNT(*) AS picks,
        SAFE_DIVIDE(COUNT(DISTINCT game) * 100.0, MAX(total.total_games)) AS pick_rate,
        COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
        COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
        SAFE_DIVIDE(
            COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) * 100.0,
            COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END)
        ) AS win_rate
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`, total
    WHERE {where_main}
    GROUP BY brawler_name
    ORDER BY picks DESC
""", params_main)

if df_brawlers.empty:
    st.warning("No data found for the selected filters.")
else:
    st.dataframe(
        df_brawlers,
        use_container_width=True,
        column_order=["brawler_img", "brawler_name", "picks", "pick_rate", "wins", "losses", "win_rate"],
        column_config={
            "brawler_img":  st.column_config.ImageColumn(""),
            "brawler_name": st.column_config.TextColumn("Brawler"),
            "picks":        st.column_config.NumberColumn("Picks",     format="%d"),
            "pick_rate":    st.column_config.ProgressColumn("Pick Rate", format="%.1f%%", min_value=0, max_value=100),
            "wins":         st.column_config.NumberColumn("Wins",      format="%d"),
            "losses":       st.column_config.NumberColumn("Losses",    format="%d"),
            "win_rate":     st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
        },
        hide_index=True
    )

    st.markdown("---")
    st.subheader("🚮 Throw List")
    st.caption("Brawlers with Pick Rate ≥ 30% but Win Rate < 45% — high trust, low delivery.")

    df_throw = (
        df_brawlers[
            (df_brawlers["pick_rate"] >= 30) &
            (df_brawlers["win_rate"]  < 45)
        ]
        .sort_values("win_rate")
        .reset_index(drop=True)
    )

    if df_throw.empty:
        st.info("No brawlers matching Throw List criteria in the current filter.")
    else:
        st.dataframe(
            df_throw,
            use_container_width=True,
            column_order=["brawler_img", "brawler_name", "picks", "pick_rate", "win_rate"],
            column_config={
                "brawler_img":  st.column_config.ImageColumn(""),
                "brawler_name": st.column_config.TextColumn("Brawler"),
                "picks":        st.column_config.NumberColumn("Picks",     format="%d"),
                "pick_rate":    st.column_config.ProgressColumn("Pick Rate", format="%.1f%%", min_value=0, max_value=100),
                "win_rate":     st.column_config.ProgressColumn("Win Rate",  format="%.1f%%", min_value=0, max_value=100),
            },
            hide_index=True
        )

# ============================================================
# COMPOSITION ANALYSIS
# ============================================================
st.markdown("---")

with st.expander("🤝 Composition Analysis", expanded=False):
    tab_trio, tab_ctr = st.tabs([
        "🔺 Trio — Full Composition",
        "🛡️ Counter — Opposite Teams",
    ])

    # ── TRIO ─────────────────────────────────────────────────
    with tab_trio:
        st.caption(
            "Three brawlers that played on the **same team** in the same match. "
            "Minimum 5 games. Sorted by win rate. "
            "Filter by **Team** in the sidebar to see your team's compositions. "
            "Click a row to see maps and teams where this composition was used."
        )

        df_trio = fetch_data(f"""
            WITH filtered AS (
                SELECT game, brawler_name, brawler_img, player_team, player_result
                FROM `brawl-sandbox.brawl_stats.vw_battles_python`
                WHERE {where_main}
            )
            SELECT
                a.brawler_name     AS brawler_a,
                MAX(a.brawler_img) AS img_a,
                b.brawler_name     AS brawler_b,
                MAX(b.brawler_img) AS img_b,
                c.brawler_name     AS brawler_c,
                MAX(c.brawler_img) AS img_c,
                COUNT(DISTINCT a.game) AS games,
                SAFE_DIVIDE(
                    COUNT(DISTINCT CASE WHEN a.player_result = 'victory' THEN a.game END) * 100.0,
                    COUNT(DISTINCT CASE WHEN a.player_result IN ('victory','defeat') THEN a.game END)
                ) AS win_rate
            FROM filtered a
            JOIN filtered b
                ON  a.game        = b.game
                AND a.player_team = b.player_team
                AND a.brawler_name < b.brawler_name
            JOIN filtered c
                ON  a.game        = c.game
                AND a.player_team = c.player_team
                AND b.brawler_name < c.brawler_name
            GROUP BY brawler_a, brawler_b, brawler_c
            HAVING games >= 5
            ORDER BY win_rate DESC, games DESC
            LIMIT 50
        """, params_main)

        if df_trio.empty:
            st.info("Not enough data to show trio compositions (minimum 5 games per composition).")
        else:
            ev_trio = st.dataframe(
                df_trio,
                use_container_width=True,
                column_order=["img_a", "brawler_a", "img_b", "brawler_b", "img_c", "brawler_c", "games", "win_rate"],
                column_config={
                    "img_a":     st.column_config.ImageColumn(""),
                    "brawler_a": st.column_config.TextColumn("Brawler A"),
                    "img_b":     st.column_config.ImageColumn(""),
                    "brawler_b": st.column_config.TextColumn("Brawler B"),
                    "img_c":     st.column_config.ImageColumn(""),
                    "brawler_c": st.column_config.TextColumn("Brawler C"),
                    "games":     st.column_config.NumberColumn("Games",    format="%d"),
                    "win_rate":  st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
                },
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            sel_trio = ev_trio.selection.rows
            if sel_trio:
                row_trio = df_trio.iloc[sel_trio[0]]
                ba = row_trio["brawler_a"]
                bb = row_trio["brawler_b"]
                bc = row_trio["brawler_c"]

                st.markdown(f"#### 🔍 Detail: **{ba}** + **{bb}** + **{bc}**")

                trio_detail_params = json.loads(params_main) + [
                    {"name": "ba", "bq_type": "STRING", "value": ba},
                    {"name": "bb", "bq_type": "STRING", "value": bb},
                    {"name": "bc", "bq_type": "STRING", "value": bc},
                ]
                df_trio_detail = fetch_data(f"""
		    WITH base AS (
		        SELECT game, brawler_name, player_team, player_result,
		               map, map_img, mode
		        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
		        WHERE {where_main}
		    ),
		    matching_games AS (
		        SELECT a.game, a.player_team
		        FROM base a
		        JOIN base b
		            ON a.game = b.game
		            AND a.player_team = b.player_team
		            AND b.brawler_name = @bb
		        JOIN base c
		            ON a.game = c.game
		            AND a.player_team = c.player_team
		            AND c.brawler_name = @bc
		        WHERE a.brawler_name = @ba
		          AND a.player_team IS NOT NULL
		    ),
		    game_info AS (
		        SELECT game, MAX(map_img) AS map_img, MAX(map) AS map,
		               MAX(mode) AS mode, MAX(player_result) AS player_result
		        FROM base
		        GROUP BY game
		    )
		    SELECT
		        gi.map_img,
		        gi.map,
		        gi.mode,
		        mg.player_team AS team,
		        COUNT(DISTINCT mg.game) AS games,
		        SAFE_DIVIDE(
		            COUNT(DISTINCT CASE WHEN gi.player_result = 'victory' THEN mg.game END) * 100.0,
		            COUNT(DISTINCT CASE WHEN gi.player_result IN ('victory','defeat') THEN mg.game END)
		        ) AS win_rate
		    FROM matching_games mg
		    JOIN game_info gi ON mg.game = gi.game
		    GROUP BY gi.map_img, gi.map, gi.mode, mg.player_team
		    ORDER BY games DESC
""", json.dumps(trio_detail_params))



                if df_trio_detail.empty:
                    st.info("No detail data found.")
                else:
                    dcol1, dcol2 = st.columns(2)

                    with dcol1:
                        st.markdown("**🗺️ By Map**")
                        st.dataframe(
                            df_trio_detail[["map_img", "map", "mode", "games", "win_rate"]].drop_duplicates().sort_values("games", ascending=False),
                            use_container_width=True,
                            column_config={
                                "map_img":  st.column_config.ImageColumn(""),
                                "map":      st.column_config.TextColumn("Map"),
                                "mode":     st.column_config.TextColumn("Mode"),
                                "games":    st.column_config.NumberColumn("Games",    format="%d"),
                                "win_rate": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
                            },
                            hide_index=True
                        )

                    with dcol2:
                        st.markdown("**🏆 By Team**")
                        df_by_team = (
                            df_trio_detail.groupby("team")
                            .agg(games=("games", "sum"), win_rate=("win_rate", "mean"))
                            .reset_index()
                            .sort_values("games", ascending=False)
                        )
                        st.dataframe(
                            df_by_team,
                            use_container_width=True,
                            column_config={
                                "team":     st.column_config.TextColumn("Team"),
                                "games":    st.column_config.NumberColumn("Games",    format="%d"),
                                "win_rate": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
                            },
                            hide_index=True
                        )
    # ── COUNTER ──────────────────────────────────────────────
    with tab_ctr:
        st.caption(
            "Pairs of brawlers that faced each other on **opposite teams** in the same match. "
            "Win Rate shows how often Brawler A's team wins. Minimum 10 games. "
            "Click a row to see maps and teams where this matchup occurred."
        )

        df_ctr = fetch_data(f"""
            WITH filtered AS (
                SELECT game, brawler_name, brawler_img, player_team, player_result
                FROM `brawl-sandbox.brawl_stats.vw_battles_python`
                WHERE {where_main}
            )
            SELECT
                a.brawler_name     AS brawler_a,
                MAX(a.brawler_img) AS img_a,
                b.brawler_name     AS brawler_b,
                MAX(b.brawler_img) AS img_b,
                COUNT(DISTINCT a.game) AS games,
                SAFE_DIVIDE(
                    COUNT(DISTINCT CASE WHEN a.player_result = 'victory' THEN a.game END) * 100.0,
                    COUNT(DISTINCT CASE WHEN a.player_result IN ('victory','defeat') THEN a.game END)
                ) AS win_rate_a
            FROM filtered a
            JOIN filtered b
                ON  a.game        = b.game
                AND a.player_team != b.player_team
                AND a.brawler_name < b.brawler_name
            GROUP BY brawler_a, brawler_b
            HAVING games >= 10
            ORDER BY win_rate_a DESC, games DESC
            LIMIT 50
        """, params_main)

        if df_ctr.empty:
            st.info("Not enough data to show counter matchups (minimum 10 games per pair).")
        else:
            ev_ctr = st.dataframe(
                df_ctr,
                use_container_width=True,
                column_order=["img_a", "brawler_a", "img_b", "brawler_b", "games", "win_rate_a"],
                column_config={
                    "img_a":      st.column_config.ImageColumn(""),
                    "brawler_a":  st.column_config.TextColumn("Brawler A"),
                    "img_b":      st.column_config.ImageColumn(""),
                    "brawler_b":  st.column_config.TextColumn("Brawler B"),
                    "games":      st.column_config.NumberColumn("Games",          format="%d"),
                    "win_rate_a": st.column_config.ProgressColumn("Win Rate (A)", format="%.1f%%", min_value=0, max_value=100),
                },
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            sel_ctr = ev_ctr.selection.rows
            if sel_ctr:
                row_ctr = df_ctr.iloc[sel_ctr[0]]
                ca = row_ctr["brawler_a"]
                cb = row_ctr["brawler_b"]

                st.markdown(f"#### 🔍 Detail: **{ca}** vs **{cb}**")

                ctr_detail_params = json.loads(params_main) + [
                    {"name": "ca", "bq_type": "STRING", "value": ca},
                    {"name": "cb", "bq_type": "STRING", "value": cb},
                ]

                df_ctr_detail = fetch_data(f"""
                    WITH base AS (
                        SELECT game, brawler_name, player_team, player_result,
                               map, map_img, mode
                        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
                        WHERE {where_main}
                    ),
                    matching_games AS (
                        SELECT a.game, a.player_team AS team_a,
                               b.player_team AS team_b, a.player_result AS result_a
                        FROM base a
                        JOIN base b
                            ON  a.game        = b.game
                            AND a.player_team != b.player_team
                            AND b.brawler_name = @cb
                        WHERE a.brawler_name = @ca
                    ),
                    game_info AS (
                        SELECT game, MAX(map_img) AS map_img,
                               MAX(map) AS map, MAX(mode) AS mode
                        FROM base
                        GROUP BY game
                    )
                    SELECT
                        gi.map_img,
                        gi.map,
                        gi.mode,
                        mg.team_a,
                        mg.team_b,
                        COUNT(DISTINCT mg.game) AS games,
                        SAFE_DIVIDE(
                            COUNT(DISTINCT CASE WHEN mg.result_a = 'victory' THEN mg.game END) * 100.0,
                            COUNT(DISTINCT CASE WHEN mg.result_a IN ('victory','defeat') THEN mg.game END)
                        ) AS win_rate_a
                    FROM matching_games mg
                    JOIN game_info gi ON mg.game = gi.game
                    GROUP BY gi.map_img, gi.map, gi.mode, mg.team_a, mg.team_b
                    ORDER BY games DESC
                """, json.dumps(ctr_detail_params))

                if df_ctr_detail.empty:
                    st.info("No detail data found.")
                else:
                    dcol1, dcol2 = st.columns(2)

                    with dcol1:
                        st.markdown("**🗺️ By Map**")
                        df_map_ctr = (
                            df_ctr_detail.groupby(["map_img", "map", "mode"])
                            .agg(games=("games", "sum"), win_rate_a=("win_rate_a", "mean"))
                            .reset_index()
                            .sort_values("games", ascending=False)
                        )
                        st.dataframe(
                            df_map_ctr,
                            use_container_width=True,
                            column_config={
                                "map_img":    st.column_config.ImageColumn(""),
                                "map":        st.column_config.TextColumn("Map"),
                                "mode":       st.column_config.TextColumn("Mode"),
                                "games":      st.column_config.NumberColumn("Games",          format="%d"),
                                "win_rate_a": st.column_config.ProgressColumn("Win Rate (A)", format="%.1f%%", min_value=0, max_value=100),
                            },
                            hide_index=True
                        )

                    with dcol2:
                        st.markdown(f"**🏆 Teams using {ca}**")
                        df_team_ctr = (
                            df_ctr_detail.groupby("team_a")
                            .agg(games=("games", "sum"), win_rate_a=("win_rate_a", "mean"))
                            .reset_index()
                            .rename(columns={"team_a": "team"})
                            .sort_values("games", ascending=False)
                        )
                        st.dataframe(
                            df_team_ctr,
                            use_container_width=True,
                            column_config={
                                "team":       st.column_config.TextColumn("Team"),
                                "games":      st.column_config.NumberColumn("Games",          format="%d"),
                                "win_rate_a": st.column_config.ProgressColumn("Win Rate (A)", format="%.1f%%", min_value=0, max_value=100),
                            },
                            hide_index=True
                        )


st.markdown("---")

# ============================================================
# PLAYERS
# ============================================================
st.header("👥 Players")

if show_only_active:
    df_players = fetch_data(f"""
        WITH stats AS (
            SELECT
                player_tag,
                ARRAY_AGG(player_name  ORDER BY battle_time DESC LIMIT 1)[OFFSET(0)] AS player_name_latest,
                ARRAY_AGG(player_img   ORDER BY battle_time DESC LIMIT 1)[OFFSET(0)] AS player_img_fact,
                ARRAY_AGG(player_team  ORDER BY battle_time DESC LIMIT 1)[OFFSET(0)] AS player_team_fact,
                COUNT(DISTINCT game) AS games,
                COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
                COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
                SAFE_DIVIDE(
                    COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) * 100.0,
                    COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END)
                ) AS win_rate
            FROM `brawl-sandbox.brawl_stats.vw_battles_python`
            WHERE {where_main}
            GROUP BY player_tag
        )
        SELECT
            COALESCE(sp.PL_LINK, s.player_img_fact)   AS player_img,
            s.player_name_latest                       AS player_name,
            COALESCE(sp.PL_CTEAM, s.player_team_fact) AS team,
            s.games, s.wins, s.losses, s.win_rate
        FROM stats s
        INNER JOIN `brawl-sandbox.brawl_stats.dim_source_players` sp
            ON s.player_tag = sp.PL_TAG AND sp.is_active = TRUE
        ORDER BY s.games DESC
    """, params_main)
else:
    df_players = fetch_data(f"""
        SELECT
            ARRAY_AGG(player_img  ORDER BY battle_time DESC LIMIT 1)[OFFSET(0)] AS player_img,
            ARRAY_AGG(player_name ORDER BY battle_time DESC LIMIT 1)[OFFSET(0)] AS player_name,
            ARRAY_AGG(player_team ORDER BY battle_time DESC LIMIT 1)[OFFSET(0)] AS team,
            COUNT(DISTINCT game) AS games,
            COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
            COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
            SAFE_DIVIDE(
                COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) * 100.0,
                COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END)
            ) AS win_rate
        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        WHERE {where_main}
        GROUP BY player_tag
        ORDER BY games DESC
    """, params_main)

if df_players.empty:
    st.warning("No data found for the selected filters.")
else:
    df_players["player_img"] = convert_img_column(df_players["player_img"])
    st.dataframe(
        df_players,
        use_container_width=True,
        column_order=["player_img", "player_name", "team", "games", "wins", "losses", "win_rate"],
        column_config={
            "player_img":  st.column_config.ImageColumn(""),
            "player_name": st.column_config.TextColumn("Player"),
            "team":        st.column_config.TextColumn("Team"),
            "games":       st.column_config.NumberColumn("Games",    format="%d"),
            "wins":        st.column_config.NumberColumn("Wins",     format="%d"),
            "losses":      st.column_config.NumberColumn("Losses",   format="%d"),
            "win_rate":    st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
        },
        hide_index=True
    )

st.markdown("---")

# ============================================================
# TEAMS
# ============================================================
st.header("🏆 Teams")

df_teams = fetch_data(f"""
    WITH filtered AS (
        SELECT game, player_team, player_result, brawler_name
        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        WHERE {where_main} AND player_team IS NOT NULL
    ),
    brawler_picks AS (
        SELECT player_team, brawler_name, COUNT(*) AS picks,
               ROW_NUMBER() OVER (PARTITION BY player_team ORDER BY COUNT(*) DESC) AS rn
        FROM filtered
        GROUP BY player_team, brawler_name
    ),
    top_brawlers AS (
        SELECT player_team,
               STRING_AGG(brawler_name, ', ' ORDER BY rn) AS top_brawlers
        FROM brawler_picks WHERE rn <= 3
        GROUP BY player_team
    )
    SELECT
        f.player_team AS team,
        COUNT(DISTINCT f.game) AS games,
        COUNT(DISTINCT CASE WHEN f.player_result = 'victory' THEN f.game END) AS wins,
        COUNT(DISTINCT CASE WHEN f.player_result = 'defeat'  THEN f.game END) AS losses,
        SAFE_DIVIDE(
            COUNT(DISTINCT CASE WHEN f.player_result = 'victory' THEN f.game END) * 100.0,
            COUNT(DISTINCT CASE WHEN f.player_result IN ('victory','defeat') THEN f.game END)
        ) AS win_rate,
        tb.top_brawlers
    FROM filtered f
    LEFT JOIN top_brawlers tb ON f.player_team = tb.player_team
    GROUP BY f.player_team, tb.top_brawlers
    ORDER BY games DESC
""", params_main)

if df_teams.empty:
    st.warning("No data found for the selected filters.")
else:
    st.dataframe(
        df_teams,
        use_container_width=True,
        column_order=["team", "games", "wins", "losses", "win_rate", "top_brawlers"],
        column_config={
            "team":         st.column_config.TextColumn("Team"),
            "games":        st.column_config.NumberColumn("Games",      format="%d"),
            "wins":         st.column_config.NumberColumn("Wins",       format="%d"),
            "losses":       st.column_config.NumberColumn("Losses",     format="%d"),
            "win_rate":     st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
            "top_brawlers": st.column_config.TextColumn("Top Brawlers", help="Top 3 most picked brawlers"),
        },
        hide_index=True
    )

st.markdown("---")

# ============================================================
# DRAFT VIEWER
# ============================================================
st.header("📋 Draft Viewer")

df_games = fetch_data(f"""
    SELECT
        game,
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', MAX(battle_time)) AS battletime,
        map,
        mode,
        MAX(map_img) AS map_img
    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
    WHERE {where_main}
    GROUP BY game, map, mode
    ORDER BY MAX(battle_time) DESC
""", params_main)

if df_games.empty:
    st.warning("No games found for the selected filters.")
else:
    st.caption("Click a row to view the draft event")
    event = st.dataframe(
        df_games,
        use_container_width=True,
        column_order=["game", "battletime", "map", "mode"],
        column_config={
            "game":       st.column_config.NumberColumn("Game", format="%d"),
            "battletime": st.column_config.TextColumn("Date / Time"),
            "map":        st.column_config.TextColumn("Map"),
            "mode":       st.column_config.TextColumn("Mode"),
        },
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    selected = event.selection.rows
    if selected:
        row = df_games.iloc[selected[0]]
        st.markdown("---")
        render_draft(int(row["game"]), row["map"], row["map_img"], row["battletime"])

st.markdown("---")

# ============================================================
# HEAD TO HEAD
# ============================================================
st.header("⚔️ Head to Head")

all_teams = st.session_state.f_team
if not all_teams:
    st.info("Select a team in the sidebar to see matchups.")
else:
    h2h_team = all_teams[0]
    where_h2h, base_params = build_where_h2h()

    if len(all_teams) > 1:
        other_teams = ", ".join(all_teams[1:])
        st.warning(
            f"Head to Head is showing results for **{h2h_team}** only. "
            f"To compare **{other_teams}** side by side, check the Timeline below."
        )

    params_h2h = json.dumps(base_params + [
        {"name": "h2h_team", "bq_type": "STRING", "value": h2h_team}
    ])

    df_h2h = fetch_data(f"""
        WITH my_games AS (
            SELECT DISTINCT game, player_team AS my_team, player_result AS my_result
            FROM `brawl-sandbox.brawl_stats.vw_battles_python`
            WHERE {where_h2h} AND player_team = @h2h_team
        ),
        opponent_games AS (
            SELECT DISTINCT game, player_team AS opp_team
            FROM `brawl-sandbox.brawl_stats.vw_battles_python`
            WHERE player_team != @h2h_team AND player_team IS NOT NULL
        )
        SELECT
            og.opp_team AS opponent,
            COUNT(DISTINCT mg.game) AS games,
            COUNT(DISTINCT CASE WHEN mg.my_result = 'victory' THEN mg.game END) AS wins,
            COUNT(DISTINCT CASE WHEN mg.my_result = 'defeat'  THEN mg.game END) AS losses,
            SAFE_DIVIDE(
                COUNT(DISTINCT CASE WHEN mg.my_result = 'victory' THEN mg.game END) * 100.0,
                COUNT(DISTINCT CASE WHEN mg.my_result IN ('victory','defeat') THEN mg.game END)
            ) AS win_rate
        FROM my_games mg
        JOIN opponent_games og ON mg.game = og.game
        GROUP BY og.opp_team
        ORDER BY games DESC
    """, params_h2h)

    if df_h2h.empty:
        st.warning("No matchups found.")
    else:
        st.subheader(f"{h2h_team} — Matchup History")
        st.caption("Click a row to see the game history against that opponent")

        ev_h2h = st.dataframe(
            df_h2h,
            use_container_width=True,
            column_order=["opponent", "games", "wins", "losses", "win_rate"],
            column_config={
                "opponent": st.column_config.TextColumn("Opponent"),
                "games":    st.column_config.NumberColumn("Games",    format="%d"),
                "wins":     st.column_config.NumberColumn("Wins",     format="%d"),
                "losses":   st.column_config.NumberColumn("Losses",   format="%d"),
                "win_rate": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=100),
            },
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        sel_opp = ev_h2h.selection.rows
        if sel_opp:
            opp_name = df_h2h.iloc[sel_opp[0]]["opponent"]
            
            # --- NOVO: FUNÇÃO CALLBACK ---
            # Essa função roda ANTES da página recarregar, evitando o erro
            def apply_h2h_filter():
                st.session_state["f_team"] = [h2h_team, opp_name]
                st.session_state["h2h_toggle_active"] = True
            # -----------------------------

            st.info(f"Selecionado: **{opp_name}**")
            
            current_teams = st.session_state.get("f_team", [])
            already_active = (
                len(current_teams) == 2 and 
                opp_name in current_teams and 
                h2h_team in current_teams and 
                st.session_state.get("h2h_toggle_active", False)
            )

            if not already_active:
                def apply_h2h_filter():
                    st.session_state["f_team"] = [h2h_team, opp_name]
                    st.session_state["h2h_toggle_active"] = True

                st.button(
                    f"🌪️ Filter Dashboard: {h2h_team} vs {opp_name}", # <--- Traduzido aqui
                    use_container_width=True,
                    type="primary",
                    on_click=apply_h2h_filter
                )

            params_matchup = json.dumps(base_params + [
                {"name": "h2h_team", "bq_type": "STRING", "value": h2h_team},
                {"name": "opp_name", "bq_type": "STRING", "value": opp_name},
            ])

            df_matchups = fetch_data(f"""
                WITH my_games AS (
                    SELECT DISTINCT game, player_team AS my_team, player_result AS my_result
                    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
                    WHERE {where_h2h} AND player_team = @h2h_team
                ),
                opp_games AS (
                    SELECT DISTINCT game, player_team AS opp_team, player_result AS opp_result
                    FROM `brawl-sandbox.brawl_stats.vw_battles_python`
                    WHERE player_team = @opp_name AND player_team IS NOT NULL
                ),
                joined AS (
                    SELECT mg.game, mg.my_team, mg.my_result,
                           og.opp_team, og.opp_result
                    FROM my_games mg JOIN opp_games og ON mg.game = og.game
                )
                SELECT
                    j.game,
                    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', MAX(v.battle_time)) AS battletime,
                    MAX(v.map)     AS map,
                    MAX(v.map_img) AS map_img,
                    MAX(v.mode)    AS mode,
                    MAX(j.my_result)  AS my_result,
                    MAX(j.opp_result) AS opp_result
                FROM joined j
                JOIN `brawl-sandbox.brawl_stats.vw_battles_python` v ON j.game = v.game
                GROUP BY j.game
                ORDER BY MAX(v.battle_time) DESC
            """, params_matchup)

            if not df_matchups.empty:
                st.markdown("---")
                st.subheader(f"{h2h_team} vs {opp_name}")
                st.caption("Click a row to view the draft")

                df_matchups["my_result_show"]  = df_matchups["my_result"].apply(
                    lambda x: "🏆 WIN" if x == "victory" else "💀 LOSS"
                )
                df_matchups["opp_result_show"] = df_matchups["opp_result"].apply(
                    lambda x: "🏆 WIN" if x == "victory" else "💀 LOSS"
                )

                ev_match = st.dataframe(
                    df_matchups,
                    use_container_width=True,
                    column_order=["game", "battletime", "map", "mode", "my_result_show", "opp_result_show"],
                    column_config={
                        "game":            st.column_config.NumberColumn("Game", format="%d"),
                        "battletime":      st.column_config.TextColumn("Date / Time"),
                        "map":             st.column_config.TextColumn("Map"),
                        "mode":            st.column_config.TextColumn("Mode"),
                        "my_result_show":  st.column_config.TextColumn(h2h_team),
                        "opp_result_show": st.column_config.TextColumn(opp_name),
                    },
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row"
                )

                sel_game = ev_match.selection.rows
                if sel_game:
                    row = df_matchups.iloc[sel_game[0]]
                    st.markdown("---")
                    render_draft(int(row["game"]), row["map"], row["map_img"], row["battletime"])

st.markdown("---")

# ============================================================
# TIMELINE
# ============================================================
st.header("📈 Timeline")

if not st.session_state.f_team:
    st.info("Select a team in the sidebar to see the timeline.")
else:
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        granularity = st.radio("Granularity", ["Day", "Week"], horizontal=True, key="tl_gran")
    with tcol2:
        wr_type = st.radio("Win Rate", ["Daily", "Cumulative"], horizontal=True, key="tl_type")

    period_expr = "DATE(battle_time)" if granularity == "Day" else "DATE_TRUNC(DATE(battle_time), WEEK)"

    df_tl = fetch_data(f"""
        SELECT
            player_team,
            {period_expr} AS period,
            COUNT(DISTINCT CASE WHEN player_result = 'victory' THEN game END) AS wins,
            COUNT(DISTINCT CASE WHEN player_result = 'defeat'  THEN game END) AS losses,
            COUNT(DISTINCT CASE WHEN player_result IN ('victory','defeat') THEN game END) AS total_games
        FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        WHERE {where_main} AND player_team IS NOT NULL
        GROUP BY player_team, period
        ORDER BY player_team, period
    """, params_main)

    if df_tl.empty:
        st.warning("No data found for the selected filters.")
    else:
        df_tl["period"] = pd.to_datetime(df_tl["period"])
        df_tl = df_tl.sort_values(["player_team", "period"])

        if wr_type == "Daily":
            df_tl["win_rate"] = (df_tl["wins"] / df_tl["total_games"] * 100).round(1)
            ylabel = "Win Rate — Daily"
        else:
            df_tl["cum_wins"]  = df_tl.groupby("player_team")["wins"].cumsum()
            df_tl["cum_games"] = df_tl.groupby("player_team")["total_games"].cumsum()
            df_tl["win_rate"]  = (df_tl["cum_wins"] / df_tl["cum_games"] * 100).round(1)
            ylabel = "Win Rate — Cumulative"

        df_tl["games_label"] = df_tl["total_games"].astype(str) + " games"

        fig = px.line(
            df_tl,
            x="period",
            y="win_rate",
            color="player_team",
            markers=True,
            hover_data={"games_label": True, "total_games": False},
            labels={"period": "Date", "win_rate": ylabel, "player_team": "Team", "games_label": "Games"},
            title=f"Win Rate over Time — {granularity} / {wr_type}"
        )
        fig.add_hline(
            y=50, line_dash="dash", line_color="gray", opacity=0.5,
            annotation_text="50%", annotation_position="bottom right"
        )
        fig.update_layout(
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            xaxis_title="Date",
            yaxis_title=ylabel,
            legend_title="Team",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
