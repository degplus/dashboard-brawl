import streamlit as st
import pandas as pd
import json
import plotly.express as px
from google.oauth2 import service_account
from google.cloud import bigquery

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Meta — DegStats", page_icon="👑", layout="wide")

# ============================================================
# 2. BIGQUERY CLIENT & AUTHENTICATION GUARD
# ============================================================
from login import check_existing_session, apply_ui_permissions
from auth import do_logout

@st.cache_resource
def get_bq_client():
    project_id = st.secrets.get("gcp_project", "brawl-sandbox")
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials, project=project_id)

client = get_bq_client()

# O Segurança da Catraca
if not check_existing_session(client):
    st.switch_page("Overview.py")

# A Camuflagem
apply_ui_permissions()

if st.session_state.get("must_change_password"):
    st.warning("⚠️ You need to set a new password before continuing.")
    st.page_link("pages/3_Change_Password.py", label="👉 Click here to set your password")
    st.stop()

# ============================================================
# 3. DATA LOADERS (Obrigatório para a Peça de Lego funcionar)
# ============================================================
@st.cache_data(ttl=3600)
def fetch_data(query: str, params_json: str = None) -> pd.DataFrame:
    bq_params = []
    if params_json:
        for p in json.loads(params_json):
            param_cls = bigquery.ArrayQueryParameter if isinstance(p["value"], list) else bigquery.ScalarQueryParameter
            bq_params.append(param_cls(p["name"], p["bq_type"], p["value"]))
    job_config = bigquery.QueryJobConfig(query_parameters=bq_params)
    return client.query(query, job_config=job_config).result().to_dataframe()

@st.cache_data(ttl=3600)
def load_dim_filters() -> pd.DataFrame:
    df = fetch_data("SELECT * FROM `brawl-sandbox.brawl_stats.dim_filters`")
    df["battle_date"] = pd.to_datetime(df["battle_date"]).dt.date
    return df

@st.cache_data(ttl=3600)
def load_player_names() -> dict:
    df = fetch_data("""
        SELECT v.player_tag, v.player_name AS display_name
        FROM (
            SELECT player_tag, player_name, ROW_NUMBER() OVER (PARTITION BY player_tag ORDER BY battle_time DESC) AS rn
            FROM `brawl-sandbox.brawl_stats.vw_battles_python`
        ) v
        INNER JOIN `brawl-sandbox.brawl_stats.dim_source_players` sp ON v.player_tag = sp.PL_TAG AND sp.is_active = TRUE
        WHERE v.rn = 1
    """)
    return dict(zip(df["player_tag"], df["display_name"]))

@st.cache_data(ttl=3600)
def load_all_player_names() -> dict:
    df = fetch_data("""
        SELECT player_tag, ARRAY_AGG(player_name ORDER BY battle_time DESC LIMIT 1)[OFFSET(0)] AS display_name
        FROM `brawl-sandbox.brawl_stats.vw_battles_python` GROUP BY player_tag
    """)
    return dict(zip(df["player_tag"], df["display_name"]))

df_dim = load_dim_filters()
player_names = load_player_names()
all_player_names = load_all_player_names()

# ============================================================
# 4. SIDEBAR & FILTERS (A NOSSA PEÇA DE LEGO!)
# ============================================================
st.logo("assets/logo.png", icon_image="assets/logo.png")

with st.sidebar:
    st.markdown(f"👤 **{st.session_state.get('user_name', '')}**")
    st.caption(st.session_state.get("user_email", ""))
    if st.session_state.get("user_email") == "android.deg@gmail.com":
        st.subheader("🛠️ Admin Panel")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        from login import clear_token
        do_logout(client, st.session_state.get("session_token", ""))
        clear_token()
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

from sidebar_filters import render_sidebar_filters
filter_data = render_sidebar_filters(df_dim, player_names, all_player_names)
where_main  = filter_data["where_main"]
params_main = filter_data["params_main"]

# ============================================================
# 5. UI PRINCIPAL — META
# ============================================================
st.title("👑 Meta & Tier List")
st.caption("Visual Tier List based on Meta Score (Win Rate × Pick Rate). Adapts automatically to your sidebar filters!")
st.markdown("---")

# Buscando os dados dos brawlers
with st.spinner("Calculating Meta Score..."):
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
    """, params_main)

if df_brawlers.empty:
    st.warning("No data found for the selected filters.")
else:
    # --- A MÁGICA DA MATEMÁTICA ---
    df_brawlers["meta_score"] = (df_brawlers["pick_rate"] / 100) * (df_brawlers["win_rate"] / 100)
    df_brawlers["percentile"] = df_brawlers["meta_score"].rank(pct=True)

    def assign_tier(p):
        if p >= 0.90:   return "👑 S"
        elif p >= 0.70: return "🟢 A"
        elif p >= 0.40: return "🟡 B"
        elif p >= 0.15: return "🟠 C"
        elif p >= 0.05: return "🔴 D"
        else:           return "💀 F"

    df_brawlers["tier"] = df_brawlers["percentile"].apply(assign_tier)
    df_brawlers = df_brawlers.sort_values(by="meta_score", ascending=False).reset_index(drop=True)

    # Cores fixas para cada Tier (Para o Gráfico e para a Tabela Visual)
    tier_colors = {
        "👑 S": "#FFD700",  # Dourado
        "🟢 A": "#4CAF50",  # Verde
        "🟡 B": "#FFEB3B",  # Amarelo
        "🟠 C": "#FF9800",  # Laranja
        "🔴 D": "#F44336",  # Vermelho
        "💀 F": "#9E9E9E"   # Cinza
    }

    # ========================================================
    # GRÁFICO DE DISPERSÃO (META MAP)
    # ========================================================
    st.subheader("🗺️ Meta Map (Pick Rate vs Win Rate)")
    st.caption("Brawlers no canto superior direito são as escolhas mais fortes e seguras.")
    
    fig = px.scatter(
        df_brawlers,
        x="pick_rate",
        y="win_rate",
        color="tier",
        hover_name="brawler_name",
        hover_data={
            "tier": False, 
            "pick_rate": ":.1f", 
            "win_rate": ":.1f", 
            "meta_score": ":.4f"
        },
        color_discrete_map=tier_colors,
        labels={"pick_rate": "Pick Rate (%)", "win_rate": "Win Rate (%)"}
    )
    
    # Linha de 50% de Win Rate
    fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.3, annotation_text="50% WR")
    
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(hovermode="closest", height=500)
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ========================================================
    # TIER LIST VISUAL (HTML/CSS)
    # ========================================================
    st.subheader("📋 Visual Tier List")
    st.caption("Brawlers classificados do melhor para o pior dentro de cada tier. Passe o mouse sobre a imagem para ver os stats.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Injetando HTML para construir as linhas
    html_tier_list = "<div style='display: flex; flex-direction: column; gap: 6px;'>"
    
    ordem_tiers = ["👑 S", "🟢 A", "🟡 B", "🟠 C", "🔴 D", "💀 F"]
    
    for tier in ordem_tiers:
        brawlers_no_tier = df_brawlers[df_brawlers["tier"] == tier]
        if brawlers_no_tier.empty:
            continue
            
        color = tier_colors.get(tier, "#FFFFFF")
        letra = tier.split(' ')[1] # Pega só a letra (S, A, B...)
        
        html_tier_list += f"""
        <div style="display: flex; background-color: #1E1E1E; border-radius: 6px; border: 1px solid #333; overflow: hidden; min-height: 80px;">
            <div style="width: 80px; min-width: 80px; background-color: {color}; color: #000; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: 900; box-shadow: 2px 0px 5px rgba(0,0,0,0.3); z-index: 10;">
                {letra}
            </div>
            <div style="display: flex; flex-wrap: wrap; padding: 10px; gap: 10px; align-items: center;">
        """
        
        for _, row in brawlers_no_tier.iterrows():
            tooltip = f"{row['brawler_name']} &#10;Win Rate: {row['win_rate']:.1f}% &#10;Pick Rate: {row['pick_rate']:.1f}%"
            html_tier_list += f"""
                <div style="position: relative; display: inline-block;">
                    <img src="{row['brawler_img']}" width="60" style="border: 2px solid {color}; border-radius: 8px; background-color: #000;" title="{tooltip}">
                </div>
            """
            
        html_tier_list += """
            </div>
        </div>
        """
        
    html_tier_list += "</div>"
    
    # Renderiza a mágica na tela!
    st.markdown(html_tier_list, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ========================================================
    # TABELA DE DADOS RAW (OPCIONAL, DENTRO DE UM EXPANDER)
    # ========================================================
    with st.expander("📊 View Raw Data Table"):
        st.dataframe(
            df_brawlers,
            use_container_width=True,
            column_order=["tier", "brawler_img", "brawler_name", "picks", "pick_rate", "wins", "losses", "win_rate"],
            column_config={
                "tier":         st.column_config.TextColumn("Tier"),
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