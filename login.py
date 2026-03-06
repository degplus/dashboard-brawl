import streamlit as st
from auth import do_login, validate_token

# ============================================================
# VERIFICA SESSÃO (Com sobrevivência ao F5 via URL)
# ============================================================
def check_existing_session(client) -> bool:
    # 1. Se a gaveta da aba já tem a confirmação, o usuário está navegando normalmente
    if st.session_state.get("authenticated"):
        return True

    # 2. Se a gaveta está vazia (ex: ele deu F5), procuramos o "Ingresso" na URL
    token = st.query_params.get("token")
    
    if not token:
        return False

    # 3. Achou o ingresso na URL! Vamos ver no banco se ele AINDA é o oficial
    # (É aqui que a gente derruba o acesso simultâneo se o token tiver mudado no banco)
    user = validate_token(client, token)
    
    if not user:
        # Se não for válido (alguém logou no lugar dele), limpamos a URL e bloqueamos
        st.query_params.clear()
        return False

    # 4. Ingresso válido! Reconstruímos a gaveta para ele continuar navegando sem senha
    st.session_state["authenticated"]        = True
    st.session_state["user_email"]           = user["email"]
    st.session_state["user_name"]            = user["display_name"]
    st.session_state["must_change_password"] = user["must_change_password"]
    st.session_state["session_token"]        = token
    return True

# ============================================================
# LOGOUT SEGURO
# ============================================================
def clear_token():
    # Limpa a gaveta e joga o ingresso da URL no lixo
    st.session_state.clear()
    st.query_params.clear()

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

            # SUCESSO! 
            # 1. Coloca o Ingresso na URL do navegador
            st.query_params["token"] = result["token"]

            # 2. Preenche a gaveta da sessão atual
            st.session_state["authenticated"]        = True
            st.session_state["user_email"]           = email
            st.session_state["user_name"]            = result["display_name"]
            st.session_state["must_change_password"] = result["must_change_password"]
            st.session_state["session_token"]        = result["token"]

            st.success("Login successful!")
            st.rerun()

    return False