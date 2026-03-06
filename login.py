import streamlit as st
import extra_streamlit_components as stx
import datetime
from auth import do_login, validate_token

COOKIE_NAME = "degstats_token"

# ============================================================
# GERENCIADOR DE COOKIES (ISOLADO E SEGURO)
# ============================================================
def get_cookie_manager():
    return stx.CookieManager()

cookie_manager = get_cookie_manager()

def clear_token():
    cookie_manager.delete(COOKIE_NAME)

# ============================================================
# VERIFICA SESSÃO (Busca na gaveta, depois na carteira/cookie)
# ============================================================
def check_existing_session(client) -> bool:
    # 1. Tenta achar o crachá rápido na gaveta da aba atual
    if st.session_state.get("authenticated"):
        return True

    # 2. Se não tem na gaveta, procura na carteira (Cookie do navegador)
    token = cookie_manager.get(COOKIE_NAME)
    if not token:
        return False

    # 3. Achou o cookie! Vamos validar no banco para ver se ainda é válido
    user = validate_token(client, token)
    if not user:
        # Se o token expirou no banco, joga o cookie fora
        cookie_manager.delete(COOKIE_NAME)
        return False

    # 4. Sucesso! Põe na gaveta para o sistema ficar rápido
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

            # SALVA O COOKIE NO NAVEGADOR (Validade de 1 dia)
            expires = datetime.datetime.now() + datetime.timedelta(days=1)
            cookie_manager.set(COOKIE_NAME, result["token"], expires_at=expires)

            st.session_state["authenticated"]        = True
            st.session_state["user_email"]           = email
            st.session_state["user_name"]            = result["display_name"]
            st.session_state["must_change_password"] = result["must_change_password"]
            st.session_state["session_token"]        = result["token"]

            st.success("Login successful!")
            st.rerun()

    return False