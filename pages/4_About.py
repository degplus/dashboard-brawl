import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from login import check_existing_session, apply_ui_permissions
from auth import do_logout

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(page_title="About — DegStats", page_icon="📖", layout="centered")

# ============================================================
# 🚫 LIMPANDO O VISUAL (Escondendo menus do Streamlit)
# ============================================================
st.markdown("""
    <style>
        /* Esconde o cabeçalho superior (Menu Hamburger, Deploy, etc) */
        [data-testid="stHeader"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        header {visibility: hidden !important;}
        
        /* Esconde o rodapé padrão do Streamlit */
        footer {visibility: hidden !important; display: none !important;}
        
        /* Ajusta o espaço em branco que sobra no topo */
        .block-container {
            padding-top: 2rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. BIGQUERY CLIENT & AUTHENTICATION GUARD
# ============================================================
@st.cache_resource
def get_bq_client():
    project_id = st.secrets.get("gcp_project", "brawl-sandbox")
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials, project=project_id)

client = get_bq_client()

if not check_existing_session(client):
    st.switch_page("Overview.py")

apply_ui_permissions()

if st.session_state.get("must_change_password"):
    st.warning("⚠️ You need to set a new password before continuing.")
    st.page_link("pages/3_Change_Password.py", label="👉 Click here to set your password")
    st.stop()

# ============================================================
# 3. SIDEBAR (User Info & Logout Only)
# ============================================================
st.logo("assets/logo.png", icon_image="assets/logo.png")

with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown(f"👤 **{st.session_state.get('user_name', '')}**")
    st.caption(st.session_state.get("user_email", ""))
    
    if st.session_state.get("user_email") == "android.deg@gmail.com":
        st.subheader("🛠️ Admin Panel")
        
    st.divider()
    
    if st.button("🚪 Logout", use_container_width=True):
        from login import clear_token
        do_logout(client, st.session_state.get("session_token", ""))
        clear_token()
        for key in list(st.session_state.keys()): 
            del st.session_state[key]
        st.rerun()

# ============================================================
# 4. MAIN UI — ABOUT & DOCUMENTATION
# ============================================================
st.image("assets/logo.png", width=150)
st.title("About DegStats")
st.markdown("Welcome to **DegStats**, your ultimate tool for Brawl Stars competitive analysis and e-sports scouting. "
            "Below you will find a quick guide on how our core metrics and features work.")
st.markdown("---")

# --- GLOSSARY EXPANDERS ---
with st.expander("🔍 How the Filters Work"):
    st.markdown("""
    The sidebar filters are the heart of DegStats, designed to give you precise control over the data:
    
    - **Smart Cascading:** The filters talk to each other! If you select a specific Map, the "Brawlers" or "Teams" dropdowns will automatically update to show *only* the ones that played on that map.
    - **Memory Vault:** Your filters travel with you. If you set up a complex filter in the *Overview* page and switch to the *Meta* page, your selections remain perfectly intact.
    - **Clear All:** The trash can button at the bottom of the sidebar instantly wipes all the memory and resets the dashboard to its default state.
    """)

with st.expander("👑 Meta Score & Tier List"):
    st.markdown("""
    **How do we define the Meta?** Instead of relying on opinions, our Tier List is purely data-driven. We calculate the **Meta Score** using a standard e-sports formula:
    
    `Meta Score = Pick Rate × Win Rate`
    
    This ensures that a Brawler is only considered "S-Tier" if they are both **highly trusted by pro players** (high Pick Rate) and **highly effective in matches** (high Win Rate).
    
    **Tier Distribution:**
    - **S Tier:** Top 10% of brawlers (The absolute best)
    - **A Tier:** Next 20% (Solid, safe picks)
    - **B Tier:** Next 30% (Situational but viable)
    - **C & D Tier:** Niche or underperforming
    - **F Tier:** Bottom 5% (Usually best avoided)
    """)

with st.expander("⚔️ Head to Head (H2H) Mode"):
    st.markdown("""
    **What is H2H Mode?** When analyzing teams, sometimes you want to know how two specific teams perform *against each other*, rather than their overall performance against everyone else.
    
    **How to use it:**
    1. Go to the **Filters** in the sidebar.
    2. Select exactly **two teams** in the Team filter.
    3. A new toggle called `⚔️ H2H Mode` will appear.
    4. Turn it **ON**. The entire dashboard will instantly recalculate to show only data from matches where Team A played directly against Team B.
    """)

with st.expander("🎖️ Player Scout & API Data"):
    st.markdown("""
    The **Player Scout** and **Player Profile** pages combine our internal competitive match database with **Live API Data**.
    
    - **Prestige Level:** A new metric showing a player's overall mastery and veteran status.
    - **Hypercharges & Buffs:** We track exactly which brawlers are fully maxed out to help you scout the true depth of a player's roster.
    """)

with st.expander("⏱️ Data Updates & Cache"):
    st.markdown("""
    **Why doesn't the data change instantly?** To keep the dashboard lightning-fast, we use a caching system. Data fetched from the database is temporarily stored in the server's memory.
    
    - **Normal Mode:** Data refreshes every 60 minutes.
    - **Tournament Mode:** Data refreshes every 10 minutes for near real-time tracking.
    
    You can always check the "Last Update" timestamp in the Overview page.
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Developed with 💙 for the Brawl Stars Competitive Community.</p>
    <p>Feedback, suggestions, or bug reports? Reach out at: <b>android.deg@gmail.com</b></p>
    <p>© 2026 DegStats. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)