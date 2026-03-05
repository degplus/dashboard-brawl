import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

# ============================================================
# CONFIGURAÇÕES (lidas do secrets.toml)
# ============================================================
def _get_cfg():
    return {
        "email":    st.secrets["smtp"]["email"],
        "password": st.secrets["smtp"]["password"],
        "name":     st.secrets["smtp"]["name"],
        "server":   "smtp.gmail.com",
        "port":     587,
    }

# ============================================================
# GERAR SENHA TEMPORÁRIA
# ============================================================
def generate_temp_password(length: int = 10) -> str:
    chars = string.ascii_letters + string.digits + "!@#$%"
    return "".join(random.choices(chars, k=length))

# ============================================================
# ENVIO BASE
# ============================================================
def _send(to_email: str, subject: str, html_body: str) -> bool:
    cfg = _get_cfg()
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{cfg['name']} <{cfg['email']}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(cfg["server"], cfg["port"]) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["email"], cfg["password"])
            server.sendmail(cfg["email"], to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Erro ao enviar email: {e}")
        return False

# ============================================================
# EMAIL — BOAS-VINDAS + SENHA TEMPORÁRIA
# ============================================================
def send_welcome_email(to_email: str, display_name: str, temp_password: str) -> bool:
    subject = "🎮 Seu acesso ao DegStats"
    body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: auto;">
        <h2 style="color: #6c63ff;">🎮 Bem-vindo ao DegStats!</h2>
        <p>Olá, <strong>{display_name}</strong>!</p>
        <p>Seu acesso foi criado. Use as credenciais abaixo para o primeiro login:</p>

        <div style="background: #f4f4f4; padding: 16px; border-radius: 8px; margin: 16px 0;">
            <p style="margin: 4px 0;">📧 <strong>Email:</strong> {to_email}</p>
            <p style="margin: 4px 0;">🔑 <strong>Senha temporária:</strong>
                <span style="font-size: 18px; font-weight: bold; color: #6c63ff;">
                    {temp_password}
                </span>
            </p>
        </div>

        <p>⚠️ Você será solicitado a <strong>criar uma nova senha</strong> no primeiro acesso.</p>
        <p style="color: #888; font-size: 12px;">
            Se você não esperava esse email, ignore-o.
        </p>
    </div>
    """
    return _send(to_email, subject, body)

# ============================================================
# EMAIL — RESET DE SENHA
# ============================================================
def send_reset_email(to_email: str, display_name: str, temp_password: str) -> bool:
    subject = "🔑 Redefinição de senha — DegStats"
    body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: auto;">
        <h2 style="color: #6c63ff;">🔑 Redefinição de Senha</h2>
        <p>Olá, <strong>{display_name}</strong>!</p>
        <p>Uma nova senha temporária foi gerada para sua conta:</p>

        <div style="background: #f4f4f4; padding: 16px; border-radius: 8px; margin: 16px 0;">
            <p style="margin: 4px 0;">🔑 <strong>Nova senha temporária:</strong>
                <span style="font-size: 18px; font-weight: bold; color: #6c63ff;">
                    {temp_password}
                </span>
            </p>
        </div>

        <p>⚠️ Você será solicitado a <strong>criar uma nova senha</strong> no próximo login.</p>
        <p style="color: #888; font-size: 12px;">
            Se você não solicitou isso, contate o administrador.
        </p>
    </div>
    """
    return _send(to_email, subject, body)
