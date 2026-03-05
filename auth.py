import bcrypt
import uuid
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# ============================================================
# CONSTANTS
# ============================================================
PROJECT  = "brawl-sandbox"
DATASET  = "brawl_stats"
TB_USERS = f"{PROJECT}.{DATASET}.dim_users"
TB_SESS  = f"{PROJECT}.{DATASET}.dim_sessions"
SESSION_DURATION_HOURS = 24

# ============================================================
# SCHEMA DEFINITIONS
# ============================================================
USERS_SCHEMA = [
    bigquery.SchemaField("email",                "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("password_hash",        "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("display_name",         "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("is_active",            "BOOL",      mode="NULLABLE"),
    bigquery.SchemaField("must_change_password", "BOOL",      mode="NULLABLE"),
    bigquery.SchemaField("created_at",           "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("updated_at",           "TIMESTAMP", mode="NULLABLE"),
]


SESSIONS_SCHEMA = [
    bigquery.SchemaField("session_token", "STRING"),
    bigquery.SchemaField("email",         "STRING"),
    bigquery.SchemaField("created_at",    "TIMESTAMP"),
    bigquery.SchemaField("last_seen",     "TIMESTAMP"),
    bigquery.SchemaField("expires_at",    "TIMESTAMP"),
    bigquery.SchemaField("is_active",     "BOOL"),
]

# ============================================================
# HELPERS — READ TABLE
# ============================================================
def _read_table(client: bigquery.Client, table_id: str) -> list[dict]:
    rows = client.query(f"SELECT * FROM `{table_id}`").result()
    return [dict(row) for row in rows]

# ============================================================
# HELPERS — WRITE TABLE (replaces entire table)
# ============================================================
def _write_table(client: bigquery.Client, table_id: str, rows: list[dict], schema: list):
    # Serialize timestamps to ISO string
    serialized = []
    for row in rows:
        r = {}
        for k, v in row.items():
            r[k] = v.isoformat() if isinstance(v, datetime) else v
        serialized.append(r)

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    client.load_table_from_json(serialized, table_id, job_config=job_config).result()

# ============================================================
# HELPERS — APPEND ROWS
# ============================================================
def _append_rows(client: bigquery.Client, table_id: str, rows: list[dict], schema: list):
    serialized = []
    for row in rows:
        r = {}
        for k, v in row.items():
            r[k] = v.isoformat() if isinstance(v, datetime) else v
        serialized.append(r)

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    client.load_table_from_json(serialized, table_id, job_config=job_config).result()

# ============================================================
# PASSWORD HELPERS
# ============================================================
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

# ============================================================
# GET USER
# ============================================================
def get_user(client: bigquery.Client, email: str) -> dict | None:
    rows = client.query(f"""
        SELECT email, password_hash, display_name, is_active, must_change_password
        FROM `{TB_USERS}`
        WHERE LOWER(email) = LOWER(@email)
        LIMIT 1
    """, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("email", "STRING", email)
    ])).result()
    result = list(rows)
    return dict(result[0]) if result else None

# ============================================================
# LOGIN
# ============================================================
def do_login(client: bigquery.Client, email: str, password: str) -> dict:
    user = get_user(client, email)

    if not user:
        return {"success": False, "message": "Email not found."}
    if not user["is_active"]:
        return {"success": False, "message": "Account disabled. Contact the administrator."}
    if not check_password(password, user["password_hash"]):
        return {"success": False, "message": "Incorrect password."}

    now        = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=SESSION_DURATION_HOURS)
    token      = str(uuid.uuid4())

    # Deactivate old sessions (B1) — read, modify, rewrite
    sessions = _read_table(client, TB_SESS)
    for s in sessions:
        if s["email"].lower() == email.lower() and s["is_active"]:
            s["is_active"] = False
    sessions.append({
        "session_token": token,
        "email":         email,
        "created_at":    now,
        "last_seen":     now,
        "expires_at":    expires_at,
        "is_active":     True,
    })
    _write_table(client, TB_SESS, sessions, SESSIONS_SCHEMA)

    return {
        "success":              True,
        "message":              "Login successful.",
        "token":                token,
        "display_name":         user["display_name"],
        "must_change_password": user["must_change_password"],
    }

# ============================================================
# VALIDATE TOKEN (F5 / reload)
# ============================================================
def validate_token(client: bigquery.Client, token: str) -> dict | None:
    now = datetime.now(timezone.utc)

    rows = client.query(f"""
        SELECT s.email, s.expires_at, u.display_name, u.must_change_password
        FROM `{TB_SESS}` s
        JOIN `{TB_USERS}` u ON LOWER(s.email) = LOWER(u.email)
        WHERE s.session_token = @token
          AND s.is_active     = TRUE
          AND s.expires_at    > @now
          AND u.is_active     = TRUE
        LIMIT 1
    """, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("token", "STRING",    token),
        bigquery.ScalarQueryParameter("now",   "TIMESTAMP", now),
    ])).result()

    result = list(rows)
    if not result:
        return None

    return {
        "email":                result[0]["email"],
        "display_name":         result[0]["display_name"],
        "must_change_password": result[0]["must_change_password"],
    }

# ============================================================
# LOGOUT
# ============================================================
def do_logout(client: bigquery.Client, token: str):
    sessions = _read_table(client, TB_SESS)
    for s in sessions:
        if s["session_token"] == token:
            s["is_active"] = False
    _write_table(client, TB_SESS, sessions, SESSIONS_SCHEMA)

# ============================================================
# CHANGE PASSWORD
# ============================================================
def change_password(client: bigquery.Client, email: str, new_password: str):
    now   = datetime.now(timezone.utc)
    users = _read_table(client, TB_USERS)
    for u in users:
        if u["email"].lower() == email.lower():
            u["password_hash"]        = hash_password(new_password)
            u["must_change_password"] = False
            u["updated_at"]           = now
    _write_table(client, TB_USERS, users, USERS_SCHEMA)

# ============================================================
# CREATE USER (used by Admin Panel)
# ============================================================
def create_user(client: bigquery.Client, email: str, display_name: str, temp_password: str):
    now = datetime.now(timezone.utc)
    _append_rows(client, TB_USERS, [{
        "email":                email,
        "password_hash":        hash_password(temp_password),
        "display_name":         display_name,
        "is_active":            True,
        "must_change_password": True,
        "created_at":           now,
        "updated_at":           now,
    }], USERS_SCHEMA)

# ============================================================
# TOGGLE USER ACTIVE (used by Admin Panel)
# ============================================================
def set_user_active(client: bigquery.Client, email: str, is_active: bool):
    now   = datetime.now(timezone.utc)
    users = _read_table(client, TB_USERS)
    for u in users:
        if u["email"].lower() == email.lower():
            u["is_active"]  = is_active
            u["updated_at"] = now
    _write_table(client, TB_USERS, users, USERS_SCHEMA)

# ============================================================
# RESET PASSWORD (used by Admin Panel)
# ============================================================
def reset_password(client: bigquery.Client, email: str, temp_password: str):
    now   = datetime.now(timezone.utc)
    users = _read_table(client, TB_USERS)
    for u in users:
        if u["email"].lower() == email.lower():
            u["password_hash"]        = hash_password(temp_password)
            u["must_change_password"] = True
            u["updated_at"]           = now
    _write_table(client, TB_USERS, users, USERS_SCHEMA)
