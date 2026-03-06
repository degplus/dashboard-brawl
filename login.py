import streamlit as st
import extra_streamlit_components as stx
import datetime
import time
from auth import do_login, validate_token

COOKIE_NAME = "degstats_token"

# ============================================================
# GERENCIADOR DE COOKIES (Sem cache, mas com Key fixa)
# ============================================================
# O 'key' fixo impede que o Streamlit crie múltiplos mensageiros
cookie_manager = stx.CookieManager(key="degstats_cookie_manager")

def clear_token():
    cookie_manager.delete(COOKIE_NAME)
    time.sleep(0.5) # Dá tempo do navegador apagar

# ============================================================
# VERIFICA SESSÃO
# ============================================================
def check_existing_session(client) -> bool:
    if st.session_state.get("authenticated"):
        return True

    # Como o CookieManager roda via React/Javascript por trás dos panos, 
    # ele precisa ser lido com segurança.
    token = cookie_manager.get(COOKIE_NAME)
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

            # 1. Entrega o crachá para o mensageiro (Validade de 1 dia)
            expires = datetime.datetime.now() + datetime.timedelta(days=1)
            cookie_manager.set(COOKIE_NAME, result["token"], expires_at=expires)

            # 2. GUARDA NA GAVETA (Imediato)
            st.session_state["authenticated"]        = True
            st.session_state["user_email"]           = email
            st.session_state["user_name"]            = result["display_name"]
            st.session_state["must_change_password"] = result["must_change_password"]
            st.session_state["session_token"]        = result["token"]

            st.success("Login successful! Redirecting...")
            
            # 3. PAUSA ESTRATÉGICA (Dá tempo do mensageiro chegar no navegador)
            time.sleep(1)
            
            # 4. Agora sim recarrega a página!
            st.rerun()

    return False