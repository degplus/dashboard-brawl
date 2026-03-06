import streamlit as st
from auth import do_login, validate_token
from streamlit_cookies_controller import CookieController

COOKIE_NAME = "degstats_token"

# ============================================================
# GERENCIADOR DE COOKIES (ISOLADO POR USUÁRIO)
# ============================================================
def get_controller():
    # Isso garante que cada aba do navegador tenha seu próprio leitor de cookies
    if "cookie_controller" not in st.session_state:
        st.session_state["cookie_controller"] = CookieController()
    return st.session_state["cookie_controller"]

# ============================================================
# FUNÇÕES DE TOKEN (Usando a gaveta particular)
# ============================================================
def get_token() -> str | None:
    return get_controller().get(COOKIE_NAME)

def set_token(token: str):
    get_controller().set(COOKIE_NAME, token, max_age=86400)  # 1 dia

def clear_token():
    get_controller().remove(COOKIE_NAME)

# ============================================================
# VERIFICA SESSÃO
# ============================================================
def check_existing_session(client) -> bool:
    if st.session_state.get("authenticated"):
        return True

    token = get_token()
    if not token:
        return False

    user = validate_token(client, token)
    if not user:
        clear_token()
        return False

    st.session_state["authenticated"]        = True
    st.session_state["user_email"]           = user["email"]
    st.session_state["user_name"]            = user["display_name"]
    st.session_state["must_change_password"] = user["must_change_password"]
    st.session_state["session_token"]        = token
    return True

# ============================================================
# RENDER LOGIN FORM
# ============================================================
def render_login(client) -> bool:
    # Esconde a barra lateral
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {display: none !important;}
            [data-testid="collapsedControl"] {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/logo.png", use_container_width=True)
        st.markdown("### 🔐 Login")
        st.markdown("---")

        with st.form("login_form"):
            email    = st.text_input("📧 Email", placeholder="your@email.com")
            password = st.text_input("🔑 Password", type="password")
            submit   = st.form_submit_button("Sign In", use_container_width=True, type="primary")

        if submit:
            if not email or not password:
                st.error("Please enter both email and password.")
                return False

            with st.spinner("Verifying credentials..."):
                result = do_login(client, email, password)

            if not result["success"]:
                st.error(result["message"])
                return False

            set_token(result["token"])

            st.session_state["authenticated"]        = True
            st.session_state["user_email"]           = email
            st.session_state["user_name"]            = result["display_name"]
            st.session_state["must_change_password"] = result["must_change_password"]
            st.session_state["session_token"]        = result["token"]

            st.success("Login successful!")
            st.rerun()

    return False