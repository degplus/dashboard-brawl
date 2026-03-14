import streamlit as st
from auth import change_password
from google.oauth2 import service_account
from google.cloud import bigquery
from login import check_existing_session, apply_ui_permissions

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Change Password — DegStats", page_icon="🔑", layout="centered")

# ============================================================
# 🚫 LIMPANDO O VISUAL (Escondendo menus do Streamlit)
# ============================================================
st.markdown("""
    <style>
        /* 1. Esconde Cabeçalho, Menu Hamburger e Toolbar */
        [data-testid="stHeader"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        header {visibility: hidden !important;}
        
        /* 2. Esconde Rodapé Padrão */
        footer {visibility: hidden !important; display: none !important;}
        
        /* 3. Esconde o botão Manage App e o Viewer Badge no canto inferior */
        .viewerBadge_container {display: none !important;}
        #viewerBadge_container {display: none !important;}
        [data-testid="stAppDeployButton"] {display: none !important;}
        .stDeployButton {display: none !important;}
        [data-testid="manage-app-button"] {display: none !important;}
        
        /* 4. Ajuste de margem para o conteúdo subir e cobrir o vazio */
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

# ============================================================
# 2. BIGQUERY CLIENT
# ============================================================
@st.cache_resource
def get_bq_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(
        credentials=credentials,
        project=st.secrets.get("gcp_project", "brawl-sandbox")
    )

client = get_bq_client()

# ============================================================
# 3. GUARD — Segurança e Camuflagem
# ============================================================
if not check_existing_session(client):
    st.switch_page("Overview.py")

# Aplica a camuflagem para esconder Admin e Menu superior!
apply_ui_permissions()


# ============================================================
# 4. INTERFACE VISUAL E LÓGICA
# ============================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("### 🔑 Set New Password")
    st.markdown("---")

    if not st.session_state.get("must_change_password"):
        st.info("Your password is already set. You can change it below if you wish.")

    with st.form("change_password_form"):
        new_password  = st.text_input("🔑 New Password",     type="password")
        confirm       = st.text_input("🔑 Confirm Password", type="password")
        submit        = st.form_submit_button("Save Password", use_container_width=True, type="primary")

    if submit:
        if not new_password or not confirm:
            st.error("Please fill in both fields.")
        elif len(new_password) < 8:
            st.error("Password must be at least 8 characters.")
        elif new_password != confirm:
            st.error("Passwords do not match.")
        else:
            change_password(client, st.session_state["user_email"], new_password)
            st.session_state["must_change_password"] = False
            st.success("✅ Password updated successfully!")
            st.page_link("Overview.py", label="👉 Go to Dashboard")