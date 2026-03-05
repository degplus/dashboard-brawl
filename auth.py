import streamlit as st
import bcrypt
import uuid
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

# ============================================================
# CONSTANTES
# ============================================================
PROJECT  = "brawl-sandbox"
DATASET  = "brawl_stats"
TB_USERS = f"`{PROJECT}.{DATASET}.dim_users`"
TB_SESS  = f"`{PROJECT}.{DATASET}.dim_sessions`"
SESSION_DURATION_HOURS = 24

# ============================================================
# HELPERS DE SENHA
# ============================================================
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

# ============================================================
# BUSCAR USUÁRIO
# ============================================================
def get_user(client: bigquery.Client, email: str) -> dict | None:
    query = f"""
        SELECT email, password_hash, display_name, is_active, must_change_password
        FROM {TB_USERS}
        WHERE LOWER(email) = LOWER(@email)
        LIMIT 1
    """
    job = client.query(query, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email)]
    ))
    rows = list(job.result())
    return dict(rows[0]) if rows else None

# ============================================================
# LOGIN — VALIDA CREDENCIAIS E CRIA SESSÃO
# ============================================================
def do_login(client: bigquery.Client, email: str, password: str) -> dict:
    """
    Retorna dict com:
      - success: bool
      - message: str
      - token: str (se success)
      - must_change_password: bool (se success)
    """
    user = get_user(client, email)

    if not user:
        return {"success": False, "message": "Email não encontrado."}

    if not user["is_active"]:
        return {"success": False, "message": "Conta desativada. Contate o administrador."}

    if not check_password(password, user["password_hash"]):
        return {"success": False, "message": "Senha incorreta."}

    # Derruba sessão anterior (B1)
    client.query(f"""
        UPDATE {TB_SESS}
        SET is_active = FALSE
        WHERE email = @email AND is_active = TRUE
    """, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email)]
    )).result()

    # Cria nova sessão
    token      = str(uuid.uuid4())
    now        = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=SESSION_DURATION_HOURS)

    client.query(f"""
        INSERT INTO {TB_SESS} (session_token, email, created_at, last_seen, expires_at, is_active)
        VALUES (@token, @email, @now, @now, @expires_at, TRUE)
    """, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("token",      "STRING",    token),
        bigquery.ScalarQueryParameter("email",      "STRING",    email),
        bigquery.ScalarQueryParameter("now",        "TIMESTAMP", now),
        bigquery.ScalarQueryParameter("expires_at", "TIMESTAMP", expires_at),
    ])).result()

    return {
        "success":              True,
        "message":              "Login realizado com sucesso.",
        "token":                token,
        "display_name":         user["display_name"],
        "must_change_password": user["must_change_password"],
    }

# ============================================================
# VALIDAR TOKEN (usado no F5 / reload)
# ============================================================
def validate_token(client: bigquery.Client, token: str) -> dict | None:
    """
    Retorna dados do usuário se o token for válido, None se não.
    Também atualiza o last_seen (heartbeat).
    """
    now = datetime.now(timezone.utc)

    query = f"""
        SELECT s.email, s.expires_at, u.display_name, u.must_change_password
        FROM {TB_SESS} s
        JOIN {TB_USERS} u ON LOWER(s.email) = LOWER(u.email)
        WHERE s.session_token = @token
          AND s.is_active = TRUE
          AND s.expires_at > @now
          AND u.is_active = TRUE
        LIMIT 1
    """
    job = client.query(query, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("token", "STRING",    token),
        bigquery.ScalarQueryParameter("now",   "TIMESTAMP", now),
    ]))
    rows = list(job.result())

    if not rows:
        return None

    # Atualiza last_seen (heartbeat silencioso)
    client.query(f"""
        UPDATE {TB_SESS}
        SET last_seen = @now
        WHERE session_token = @token
    """, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("now",   "TIMESTAMP", now),
        bigquery.ScalarQueryParameter("token", "STRING",    token),
    ])).result()

    return {
        "email":                rows[0]["email"],
        "display_name":         rows[0]["display_name"],
        "must_change_password": rows[0]["must_change_password"],
    }

# ============================================================
# LOGOUT
# ============================================================
def do_logout(client: bigquery.Client, token: str):
    client.query(f"""
        UPDATE {TB_SESS}
        SET is_active = FALSE
        WHERE session_token = @token
    """, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("token", "STRING", token)]
    )).result()

# ============================================================
# TROCAR SENHA
# ============================================================
def change_password(client: bigquery.Client, email: str, new_password: str):
    new_hash = hash_password(new_password)
    now      = datetime.now(timezone.utc)
    client.query(f"""
        UPDATE {TB_USERS}
        SET password_hash = @hash,
            must_change_password = FALSE,
            updated_at = @now
        WHERE LOWER(email) = LOWER(@email)
    """, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("hash",  "STRING",    new_hash),
        bigquery.ScalarQueryParameter("now",   "TIMESTAMP", now),
        bigquery.ScalarQueryParameter("email", "STRING",    email),
    ])).result()
