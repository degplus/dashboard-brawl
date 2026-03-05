import streamlit as st
from auth import change_password, validate_token
from login import get_token, clear_token

st.set_page_config(page_title="Change Password — DegStats", page_icon="🔑", layout="centered")

# ============================================================
# GUARD — must be logged in
# ============================================================
token = get_token()
if not token:
    st.error("Session not found. Please log in again.")
    st.stop()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("### 🔑 Set New Password")
    st.markdown("---")

    if not st.session_state.get("must_change_password"):
        st.info("Your password is already set. You can change it below if you wish.")

    with st.form("change_password_form"):
        new_password  = st.text_input("🔑 New Password",     type="password")
        confirm       = st.text_input("🔑 Confirm Password", type="password")
        submit        = st.form_submit_button("Save Password", use_container_width=True, type="primary")

    if submit:
        if not new_password or not confirm:
            st.error("Please fill in both fields.")
        elif len(new_password) < 8:
            st.error("Password must be at least 8 characters.")
        elif new_password != confirm:
            st.error("Passwords do not match.")
        else:
            from google.oauth2 import service_account
            from google.cloud import bigquery

            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            client = bigquery.Client(
                credentials=credentials,
                project=st.secrets.get("gcp_project", "brawl-sandbox")
            )

            change_password(client, st.session_state["user_email"], new_password)
            st.session_state["must_change_password"] = False
            st.success("✅ Password updated successfully!")
            st.page_link("Overview.py", label="👉 Go to Dashboard")
