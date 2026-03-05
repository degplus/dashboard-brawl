import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from auth import set_user_active, reset_password, create_user
from email_sender import send_welcome_email, send_reset_email, generate_temp_password
from login import get_token

st.set_page_config(page_title="Admin Panel — DegStats", page_icon="🛠️", layout="centered")

# ============================================================
# GUARD — admin only
# ============================================================
token = get_token()
if not token:
    st.error("Session not found. Please log in again.")
    st.stop()

if st.session_state.get("user_email") != "android.deg@gmail.com":
    st.error("⛔ Access restricted to administrators.")
    st.stop()

# ============================================================
# BQ CLIENT
# ============================================================
@st.cache_resource
def get_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(
        credentials=creds,
        project=st.secrets.get("gcp_project", "brawl-sandbox")
    )

client = get_client()

# ============================================================
# LOAD USERS
# ============================================================
@st.cache_data(ttl=30)
def load_users():
    rows = client.query("""
        SELECT email, display_name, is_active, must_change_password, created_at
        FROM `brawl-sandbox.brawl_stats.dim_users`
        ORDER BY created_at DESC
    """).result()
    return [dict(r) for r in rows]

# ============================================================
# UI
# ============================================================
st.title("🛠️ Admin Panel")
st.markdown("---")

tab1, tab2 = st.tabs(["👥 Users", "➕ Add User"])

# ============================================================
# TAB 1 — USER LIST
# ============================================================
with tab1:
    st.subheader("Registered Users")
    users = load_users()

    if not users:
        st.info("No users registered yet.")
    else:
        for u in users:
            with st.expander(f"{'✅' if u['is_active'] else '❌'}  {u['display_name']}  —  {u['email']}"):
                col1, col2 = st.columns(2)

                with col1:
                    new_status = st.toggle(
                        "Active",
                        value=bool(u["is_active"]),
                        key=f"active_{u['email']}"
                    )
                    if new_status != u["is_active"]:
                        with st.spinner("Updating..."):
                            set_user_active(client, u["email"], new_status)
                            st.cache_data.clear()
                        st.rerun()

                with col2:
                    if st.button("🔄 Reset Password", key=f"reset_{u['email']}"):
                        with st.spinner("Resetting..."):
                            temp = generate_temp_password()
                            reset_password(client, u["email"], temp)
                            send_reset_email(u["email"], u["display_name"], temp)
                            st.cache_data.clear()
                        st.success(f"✅ Password reset and sent to {u['email']}")

# ============================================================
# TAB 2 — ADD USER
# ============================================================
with tab2:
    st.subheader("Add New User")

    with st.form("add_user_form"):
        new_email = st.text_input("📧 Email")
        new_name  = st.text_input("👤 Display Name")
        submit    = st.form_submit_button(
            "➕ Create & Send Welcome Email",
            use_container_width=True,
            type="primary"
        )

    if submit:
        if not new_email or not new_name:
            st.error("Please fill in all fields.")
        else:
            with st.spinner("Creating user..."):
                temp = generate_temp_password()
                create_user(client, new_email, new_name, temp)
