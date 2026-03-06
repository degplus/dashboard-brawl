import streamlit as st
from auth import do_login

# ============================================================
# VERIFICA SESSÃO (100% Isolado por Aba)
# ============================================================
def check_existing_session(client) -> bool:
    # Lê apenas a gaveta particular desta aba exata do navegador
    return st.session_state.get("authenticated", False)

# ============================================================
# LOGOUT SEGURA
# ============================================================
def clear_token():
    # Esvazia a gaveta inteira e joga a chave fora
    st.session_state.clear()

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

            # GUARDA NA GAVETA DA ABA ATUAL (Segurança máxima)
            st.session_state["authenticated"]        = True
            st.session_state["user_email"]           = email
            st.session_state["user_name"]            = result["display_name"]
            st.session_state["must_change_password"] = result["must_change_password"]
            st.session_state["session_token"]        = result["token"]

            st.rerun()

    return False