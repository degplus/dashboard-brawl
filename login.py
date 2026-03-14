import streamlit as st
from auth import do_login, validate_token, get_user, reset_password
from email_sender import generate_temp_password, send_reset_email

# ============================================================
# VERIFICA SESSÃO (À prova de F5 e Navegação)
# ============================================================
def check_existing_session(client) -> bool:
    # 1. Usuário já está navegando (mudou de página)
    if st.session_state.get("authenticated"):
        # O TRUQUE MÁGICO: Devolve o token para a URL para não sumir ao mudar de página!
        if "session_token" in st.session_state:
            st.query_params["token"] = st.session_state["session_token"]
        return True

    # 2. Usuário deu F5 (memória apagou), busca o ingresso na URL
    token = st.query_params.get("token")
    if not token:
        return False

    # 3. Valida no banco de dados (É aqui que derruba o espertinho do acesso duplo!)
    user = validate_token(client, token)
    if not user:
        st.query_params.clear()
        return False

    # 4. Ingresso válido! Restaura a gaveta da sessão
    st.session_state["authenticated"]        = True
    st.session_state["user_email"]           = user["email"]
    st.session_state["user_name"]            = user["display_name"]
    st.session_state["must_change_password"] = user["must_change_password"]
    st.session_state["session_token"]        = token

    st.query_params["token"] = token # Garante que fique na URL
    return True

# ============================================================
# CAMUFLAGEM DE UI (Esconde itens para não-admins)
# ============================================================
def apply_ui_permissions():
    # O seu código original que esconde o Admin continua aqui em cima:
    if st.session_state.get("user_email") != "android.deg@gmail.com":
        st.markdown("""
            <style>
                a[href*="Admin"] {
                    display: none !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
    # ==========================================
    # --- NOVO: SEPARADOR VISUAL DO MENU ---
    # ==========================================
    st.markdown("""
        <style>
            /* Procura exatamente o item do menu que leva ao "Change_Password" 
               e desenha uma linha suave acima dele */
            [data-testid="stSidebarNav"] li:has(a[href*="Change_Password"]) {
                border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
                margin-top: 15px !important;
                padding-top: 10px !important;
            }
        </style>
    """, unsafe_allow_html=True)

# ============================================================
# LOGOUT SEGURO
# ============================================================
def clear_token():
    st.session_state.clear()
    st.query_params.clear()

# ============================================================
# RENDER LOGIN FORM (Continua igual...)
# ============================================================
def render_login(client) -> bool:
    # Esconde a barra lateral na tela de login
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

            # Coloca na URL
            st.query_params["token"] = result["token"]

            # Coloca na Gaveta
            st.session_state["authenticated"]        = True
            st.session_state["user_email"]           = email
            st.session_state["user_name"]            = result["display_name"]
            st.session_state["must_change_password"] = result["must_change_password"]
            st.session_state["session_token"]        = result["token"]

            st.success("Login successful!")
            st.rerun()
        # ============================================================
        # NOVO: FORGOT PASSWORD (Adicione logo abaixo do fluxo de login)
        # ============================================================
        st.markdown("---")
        with st.expander("Forgot Password? 🔑"):
            st.markdown("Enter your registered email to receive a temporary password.")
            
            with st.form("forgot_password_form"):
                forgot_email = st.text_input("Email Address")
                submit_forgot = st.form_submit_button("Send Reset Email")
                
                if submit_forgot:
                    if not forgot_email:
                        st.error("Please enter an email address.")
                    else:
                        with st.spinner("Processing..."):
                            # 1. Checa se o usuário existe na base e está ativo
                            user = get_user(client, forgot_email)
                            
                            if user and user.get("is_active"):
                                # 2. Gera a senha provisória
                                temp_pw = generate_temp_password()
                                # 3. Atualiza o banco (Sua função reset_password já seta must_change_password=True!)
                                reset_password(client, forgot_email, temp_pw)
                                # 4. Dispara o email
                                send_reset_email(forgot_email, user.get("display_name", "Player"), temp_pw)
                            
                            # Dica de Segurança: Sempre mostramos a mesma mensagem de sucesso, 
                            # mesmo se o email não existir, para evitar que hackers descubram quem tem conta.
                            st.success("✅ If this email is registered and active, a recovery link was sent to your inbox.")

    return False