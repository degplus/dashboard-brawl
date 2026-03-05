import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from auth import hash_password
from email_sender import send_welcome_email, send_reset_email, generate_temp_password
from login import get_token
from datetime import datetime, timezone

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

PROJECT  = "brawl-sandbox"
DATASET  = "brawl_stats"
TB_USERS = f"`{PROJECT}.{DATASET}.dim_users`"

# ============================================================
# LOAD USERS
# ============================================================
def load_users():
    rows = client.query(f"""
        SELECT email, display_name, is_active, must_change_password, created_at
        FROM {TB_USERS}
        ORDER BY created_at DESC
    """).result()
    return list(rows)

# ============================================================
# UI
# ============================================================
st.title("🛠️ Admin Panel")
st.markdown("---")

# ── TAB LAYOUT ───────────────────────────────────────────────
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
                    # Toggle active/inactive
                    new_status = st.toggle(
                        "Active",
                        value=u["is_active"],
                        key=f"active_{u['email']}"
                    )
                    if new_status != u["is_active"]:
                        now = datetime.now(timezone.utc)
                        client.query(f"""
                            UPDATE {TB_USERS}
                            SET is_active = @status, updated_at = @now
                            WHERE email = @email
                        """, job_config=bigquery.QueryJobConfig(query_parameters=[
                            bigquery.ScalarQueryParameter("status", "BOOL",      new_status),
                            bigquery.ScalarQueryParameter("now",    "TIMESTAMP", now),
                            bigquery.ScalarQueryParameter("email",  "STRING",    u["email"]),
                        ])).result()
                        st.rerun()

                with col2:
                    # Reset password
                    if st.button("🔄 Reset Password", key=f"reset_{u['email']}"):
                        temp = generate_temp_password()
                        now  = datetime.now(timezone.utc)
                        client.query(f"""
                            UPDATE {TB_USERS}
                            SET password_hash = @hash,
                                must_change_password = TRUE,
                                updated_at = @now
                            WHERE email = @email
                        """, job_config=bigquery.QueryJobConfig(query_parameters=[
                            bigquery.ScalarQueryParameter("hash",  "STRING",    hash_password(temp)),
                            bigquery.ScalarQueryParameter("now",   "TIMESTAMP", now),
                            bigquery.ScalarQueryParameter("email", "STRING",    u["email"]),
                        ])).result()
                        send_reset_email(u["email"], u["display_name"], temp)
                        st.success(f"✅ Password reset and sent to {u['email']}")

# ============================================================
# TAB 2 — ADD USER
# ============================================================
with tab2:
    st.subheader("Add New User")

    with st.form("add_user_form"):
        new_email = st.text_input("📧 Email")
        new_name  = st.text_input("👤 Display Name")
        submit    = st.form_submit_button("➕ Create & Send Welcome Email",
                                          use_container_width=True,
                                          type="primary")

    if submit:
        if not new_email or not new_name:
            st.error("Please fill in all fields.")
        else:
            temp = generate_temp_password()
            now  = datetime.now(timezone.utc)
            client.query(f"""
                INSERT INTO {TB_USERS} (email, password_hash, display_name, is_active, must_change_password, created_at, updated_at)
                VALUES (@email, @hash, @name, TRUE, TRUE, @now, @now)
            """, job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("email", "STRING",    new_email),
                bigquery.ScalarQueryParameter("hash",  "STRING",    hash_password(temp)),
                bigquery.ScalarQueryParameter("name",  "STRING",    new_name),
                bigquery.ScalarQueryParameter("now",   "TIMESTAMP", now),
            ])).result()
            send_welcome_email(new_email, new_name, temp)
            st.success(f"✅ User **{new_name}** created! Welcome email sent to {new_email}.")
