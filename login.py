import streamlit as st
from auth import do_login, validate_token
from streamlit_cookies_controller import CookieController

COOKIE_NAME = "degstats_token"
controller  = CookieController()

# ============================================================
# LÊ O TOKEN DO COOKIE DO NAVEGADOR
# ============================================================
def get_token() -> str | None:
    return controller.get(COOKIE_NAME)

# ============================================================
# SALVA O TOKEN NO COOKIE DO NAVEGADOR
# ============================================================
def set_token(token: str):
    controller.set(COOKIE_NAME, token, max_age=86400)  # 1 dia em segundos

# ============================================================
# REMOVE O TOKEN DO COOKIE
# ============================================================
def clear_token():
    controller.remove(COOKIE_NAME)

# ============================================================
# VERIFICA SE JÁ EXISTE SESSÃO VÁLIDA (usado no F5/reload)
# ============================================================
def check_existing_session(client) -> bool:
    """
    Retorna True se o usuário já está autenticado via cookie.
    Popula st.session_state com os dados do usuário.
    """
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
# RENDERIZA O FORMULÁRIO DE LOGIN
# ============================================================
def render_login(client) -> bool:
    """
    Exibe a tela de login.
    Retorna True quando o login for bem-sucedido.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/logo.png", use_container_width=True)
        st.markdown("### 🔐 Login")
        st.markdown("---")

        with st.form("login_form"):
            email    = st.text_input("📧 Email", placeholder="seu@email.com")
            password = st.text_input("🔑 Senha", type="password")
            submit   = st.form_submit_button("Entrar", use_container_width=True, type="primary")

        if submit:
            if not email or not password:
                st.error("Preencha email e senha.")
                return False

            with st.spinner("Verificando..."):
                result = do_login(client, email, password)

            if not result["success"]:
                st.error(result["message"])
                return False

            # Salva token no cookie do navegador
            set_token(result["token"])

            # Popula session_state
            st.session_state["authenticated"]        = True
            st.session_state["user_email"]           = email
            st.session_state["user_name"]            = result["display_name"]
            st.session_state["must_change_password"] = result["must_change_password"]
            st.session_state["session_token"]        = result["token"]

            st.success("Login realizado!")
            st.rerun()

    return False
