# pages/4_Admin.py
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from google.oauth2 import service_account
from google.cloud import bigquery
from auth import set_user_active, reset_password, create_user, delete_user, update_user_expiration
from email_sender import send_welcome_email, send_reset_email, generate_temp_password

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Admin Panel — DegStats", page_icon="🛠️", layout="wide")

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
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(
        credentials=creds,
        project=st.secrets.get("gcp_project", "brawl-sandbox")
    )

client = get_bq_client()

# ============================================================
# 3. GUARD DUPLO (Logado + Administrador)
# ============================================================
from login import check_existing_session

if not check_existing_session(client):
    st.switch_page("Overview.py")

if st.session_state.get("user_email") != "android.deg@gmail.com":
    st.error("⛔ Access restricted to administrators.")
    st.stop()

# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data(ttl=30)
def load_users():
    rows = client.query("""
        SELECT email, display_name, is_active, must_change_password, created_at, expiration_date
        FROM `brawl-sandbox.brawl_stats.dim_users`
        ORDER BY created_at DESC
    """).result()
    return [dict(r) for r in rows]

@st.cache_data(ttl=30)
def load_active_sessions():
    now = datetime.now(timezone.utc)
    rows = client.query(f"""
        SELECT u.display_name, s.email, s.created_at AS login_time, s.expires_at
        FROM `brawl-sandbox.brawl_stats.dim_sessions` s
        JOIN `brawl-sandbox.brawl_stats.dim_users` u ON LOWER(s.email) = LOWER(u.email)
        WHERE s.is_active = TRUE 
          AND s.expires_at > @now
        ORDER BY s.created_at DESC
    """, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("now", "TIMESTAMP", now)
    ])).result()
    return [dict(r) for r in rows]

# ============================================================
# UI
# ============================================================
st.title("🛠️ Admin Panel")
st.markdown("---")

tab_online, tab_users, tab_add = st.tabs(["🟢 Online Now", "👥 Users Mgmt", "➕ Add User"])

# ============================================================
# TAB 1 — ONLINE NOW (O Radar)
# ============================================================
with tab_online:
    st.subheader("Active Sessions")
    st.caption("Users currently logged in. A session lasts for 24 hours.")
    
    if st.button("🔄 Refresh Radar"):
        st.cache_data.clear()
        st.rerun()
        
    sessions = load_active_sessions()
    
    if not sessions:
        st.info("No active sessions right now.")
    else:
        df_sess = pd.DataFrame(sessions)
        
        # 1. Ajuste de Fuso Horário (De UTC para Brasília GMT-3)
        # O BigQuery retorna datetime 'naive' (sem fuso na variável) mas sabemos que é UTC.
        # Então dizemos ao Pandas "Isso é UTC", e depois "Converta para São Paulo".
        # Como o BQ já entrega com fuso (tz-aware), nós apenas convertemos!
        if not df_sess.empty:
            df_sess['login_time'] = pd.to_datetime(df_sess['login_time']).dt.tz_convert('America/Sao_Paulo')
            df_sess['expires_at'] = pd.to_datetime(df_sess['expires_at']).dt.tz_convert('America/Sao_Paulo')

        # 2. Truque do Corte (Garante que só vai pra tela o que a gente quer)
        colunas_visiveis_sess = ["display_name", "email", "login_time", "expires_at"]
        df_sess_view = df_sess[colunas_visiveis_sess]

        st.dataframe(
            df_sess_view,
            use_container_width=True,
            column_config={
                "display_name": "Name",
                "email":        "Email",
                # Mudamos o texto do cabeçalho para (BRT) - Brazilian Time
                "login_time":   st.column_config.DatetimeColumn("Logged in at (BRT)", format="YYYY-MM-DD HH:mm"),
                "expires_at":   st.column_config.DatetimeColumn("Session Expiry (BRT)", format="YYYY-MM-DD HH:mm"),
            },
            hide_index=True
        )

# ============================================================
# TAB 2 — USER LIST
# ============================================================
with tab_users:
    users = load_users()

    if not users:
        st.info("No users registered yet.")
    else:
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_term = st.text_input("🔍 Search Name or Email", placeholder="Type to search...")
        with col_filter:
            status_filter = st.selectbox("Status Filter", ["All", "Active", "Inactive"])

        filtered_users = []
        for u in users:
            match_text = True
            match_status = True

            if search_term:
                term = search_term.lower()
                if term not in u["email"].lower() and term not in u["display_name"].lower():
                    match_text = False
            
            if status_filter == "Active" and not u["is_active"]:
                match_status = False
            elif status_filter == "Inactive" and u["is_active"]:
                match_status = False

            if match_text and match_status:
                filtered_users.append(u)

        st.markdown(f"*Showing **{len(filtered_users)}** user(s)*")
        st.markdown("<br>", unsafe_allow_html=True)

        if not filtered_users:
            st.warning("No users found matching your filters.")
        else:
            for u in filtered_users:
                # Mostra o status do plano no título
                exp_text = f" (Expires: {u['expiration_date']})" if u.get('expiration_date') else " (Lifetime)"
                icon = '✅' if u['is_active'] else '❌'
                
                with st.expander(f"{icon}  {u['display_name']}  —  {u['email']} {exp_text}"):
                    c_status, c_pwd, c_plan, c_del = st.columns(4)

                    # 1. Botão Status
                    with c_status:
                        new_status = st.toggle("Active Login", value=bool(u["is_active"]), key=f"active_{u['email']}")
                        if new_status != u["is_active"]:
                            with st.spinner("Updating status..."):
                                set_user_active(client, u["email"], new_status)
                                st.cache_data.clear()
                            st.rerun()

                    # 2. Botão Senha
                    with c_pwd:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Reset Password", key=f"reset_{u['email']}", use_container_width=True):
                            with st.spinner("Resetting..."):
                                temp = generate_temp_password()
                                reset_password(client, u["email"], temp)
                                send_reset_email(u["email"], u["display_name"], temp)
                                st.cache_data.clear()
                            st.success(f"Sent to {u['email']}")

                    # 3. Alterar Validade (O Crachá)
                    with c_plan:
                        new_exp = st.date_input(
                            "Subscription Expiry", 
                            value=u.get("expiration_date"),
                            key=f"exp_{u['email']}"
                        )
                        if st.button("💾 Save Date", key=f"savedate_{u['email']}", use_container_width=True):
                            with st.spinner("Saving..."):
                                update_user_expiration(client, u["email"], new_exp)
                                st.cache_data.clear()
                            st.success("Date updated!")
                            st.rerun()

                    # 4. Excluir Usuário
                    with c_del:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🗑️ Delete User", type="primary", key=f"del_{u['email']}", use_container_width=True):
                            st.session_state[f"confirm_del_{u['email']}"] = True
                            
                        # Lógica de dupla confirmação para evitar cliques acidentais
                        if st.session_state.get(f"confirm_del_{u['email']}", False):
                            st.warning("Are you absolutely sure?")
                            c_yes, c_no = st.columns(2)
                            if c_yes.button("✔️ Yes, Delete", key=f"yes_{u['email']}", use_container_width=True):
                                with st.spinner("Deleting forever..."):
                                    delete_user(client, u["email"])
                                    st.cache_data.clear()
                                st.success("User deleted!")
                                st.rerun()
                            if c_no.button("❌ Cancel", key=f"no_{u['email']}", use_container_width=True):
                                st.session_state[f"confirm_del_{u['email']}"] = False
                                st.rerun()

# ============================================================
# TAB 3 — ADD USER
# ============================================================
with tab_add:
    st.subheader("Add New User")

    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_email = st.text_input("📧 Email")
            new_name  = st.text_input("👤 Display Name")
        with col2:
            new_exp = st.date_input("📅 Expiration Date (Optional)", value=None, help="Leave empty for lifetime access.")
            
        submit = st.form_submit_button("➕ Create & Send Welcome Email", use_container_width=True, type="primary")

    if submit:
        if not new_email or not new_name:
            st.error("Please fill in email and name.")
        else:
            with st.spinner("Creating user..."):
                temp = generate_temp_password()
                create_user(client, new_email, new_name, temp, new_exp)
                send_welcome_email(new_email, new_name, temp)
                st.cache_data.clear()
            st.success(f"✅ User **{new_name}** created! Welcome email sent to {new_email}.")