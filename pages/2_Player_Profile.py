# pages/2_Tag_Lookup.py
import streamlit as st
import requests
import base64
from PIL import Image
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Player Profile — DegStats",
    page_icon="🏷️"
    layout="wide"
)

with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("---")

# ============================================================
# BRAWL STARS API
# ============================================================
BRAWL_API_BASE = "https://bsproxy.royaleapi.dev/v1"

@st.cache_data(ttl=300)
def fetch_api_profile(tag: str) -> dict | None:
    token = st.secrets.get("BRAWL_API_TOKEN", "")
    if not token:
        return {"_error": "no_token"}
    encoded = tag.strip().upper().replace("#", "%23")
    if not encoded.startswith("%23"):
        encoded = "%23" + encoded
    url = f"{BRAWL_API_BASE}/players/{encoded}"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=8
        )
        if resp.status_code == 200:
            return resp.json()
        return {"_error": f"http_{resp.status_code}", "_detail": resp.text[:300]}
    except Exception as e:
        return {"_error": "exception", "_detail": str(e)}

@st.cache_data(ttl=86400)
def img_to_base64(url: str) -> str | None:
    if not url:
        return None
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if resp.status_code == 200:
            mime = resp.headers.get("Content-Type", "image/png").split(";")[0]
            b64 = base64.b64encode(resp.content).decode()
            return f"data:{mime};base64,{b64}"
    except Exception:
        return None

# ============================================================
# HEADER
# ============================================================
st.title("🏷️ Tag Lookup")
st.caption("Search any Brawl Stars player by tag — live data from the official API.")
st.markdown("---")

# ============================================================
# INPUT
# ============================================================
col_input, col_btn = st.columns([3, 1])
with col_input:
    raw_tag = st.text_input(
        "🔎 Player Tag",
        placeholder="#8Y98Q8U",
        help="Enter the player tag with or without #"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    search = st.button("Search", use_container_width=True, type="primary")

if not raw_tag and not search:
    st.info("Enter a player tag above and click **Search**.")
    st.stop()

if not raw_tag:
    st.warning("Please enter a tag before searching.")
    st.stop()

# ============================================================
# FETCH
# ============================================================
with st.spinner("Fetching player data..."):
    profile = fetch_api_profile(raw_tag)

# ============================================================
# ERROR HANDLING
# ============================================================
if not profile:
    st.error("No data returned.")
    st.stop()

if "_error" in profile:
    err    = profile["_error"]
    detail = profile.get("_detail", "")
    if err == "no_token":
        st.error("BRAWL_API_TOKEN not found in secrets.")
    elif err == "http_404":
        st.error(f"Player **{raw_tag}** not found. Check the tag and try again.")
    elif err == "http_403":
        st.error("Access denied — check your API token or IP whitelist.")
    else:
        st.error(f"API error: {err} — {detail}")
    st.stop()

# ============================================================
# PLAYER CARD
# ============================================================
club     = profile.get("club") or {}
icon_id  = (profile.get("icon") or {}).get("id")
icon_url = f"https://cdn.brawlify.com/profile-icons/regular/{icon_id}.png" if icon_id else None

name_color_raw = profile.get("nameColor", "0xffffffff")
name_color     = "#" + name_color_raw.replace("0xff", "").replace("0x", "")

is_champion = profile.get("isQualifiedFromChampionshipChallenge", False)
tag_clean   = profile.get("tag", raw_tag)
club_name   = club.get("name", "—") if club else "—"
club_tag    = club.get("tag", "")   if club else ""

card_img, card_info = st.columns([1, 7])
with card_img:
    if icon_url:
        b64 = img_to_base64(icon_url)
        if b64:
            st.image(b64, width=80)
with card_info:
    champion_badge = (
        "&nbsp;<span style='background:#f0a500;color:#000;"
        "padding:2px 8px;border-radius:10px;font-size:0.75rem'>"
        "🏅 Championship Qualified</span>"
        if is_champion else ""
    )
    st.markdown(
        f"<span style='"
        f"color:{name_color};"
        f"font-size:1.6rem;"
        f"font-weight:bold;"
        f"text-shadow: 1px 1px 3px #000000, -1px -1px 3px #000000,"
        f"             1px -1px 3px #000000, -1px 1px 3px #000000'>"
        f"{profile.get('name', raw_tag)}</span>{champion_badge}",
        unsafe_allow_html=True
    )
    st.caption(
        f"**Tag:** `{tag_clean}`  |  "
        f"**Club:** {club_name} `{club_tag}`  |  "
        f"**Exp Level:** {profile.get('expLevel', '—')} "
        f"({profile.get('expPoints', 0):,} XP)"
    )

st.markdown("---")

# ============================================================
# KPIs
# ============================================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🏆 Trophies",         f"{profile.get('trophies', 0):,}")
k2.metric("📈 Highest Trophies", f"{profile.get('highestTrophies', 0):,}")
k3.metric("🎮 3v3 Victories",    f"{profile.get('3vs3Victories', 0):,}")
k4.metric("🥊 Solo Victories",   f"{profile.get('soloVictories', 0):,}")
k5.metric("👥 Duo Victories",    f"{profile.get('duoVictories', 0):,}")

st.markdown("---")

# ============================================================
# BRAWLER ROSTER
# ============================================================
brawlers_api = profile.get("brawlers", [])

if brawlers_api:
    st.header(f"🔓 Brawler Roster — {len(brawlers_api)} unlocked")

    search_brawler = st.text_input(
        "🔍 Filter brawler",
        placeholder="e.g. Shelly",
        key="lu_search"
    )

    df_brawlers = pd.DataFrame([
        {
            "brawler_name":     b.get("name", ""),
            "power":            b.get("power", 0),
            "rank":             b.get("rank", 0),
            "trophies":         b.get("trophies", 0),
            "highest_trophies": b.get("highestTrophies", 0),
            "max_win_streak":   b.get("maxWinStreak", 0),
            "gadgets":          len(b.get("gadgets", [])),
            "star_powers":      len(b.get("starPowers", [])),
            "gears":            len(b.get("gears", [])),
        }
        for b in brawlers_api
    ]).sort_values("trophies", ascending=False).reset_index(drop=True)

    if search_brawler:
        df_brawlers = df_brawlers[
            df_brawlers["brawler_name"].str.contains(search_brawler, case=False)
        ].reset_index(drop=True)

    max_trophies = int(df_brawlers["trophies"].max()) + 1 if not df_brawlers.empty else 1000

    st.caption(f"Showing {len(df_brawlers)} of {len(brawlers_api)} brawlers")

    st.dataframe(
        df_brawlers,
        use_container_width=True,
        column_config={
            "brawler_name":     st.column_config.TextColumn("Brawler"),
            "power":            st.column_config.NumberColumn("Power",        format="%d"),
            "rank":             st.column_config.NumberColumn("Rank",         format="%d"),
            "trophies":         st.column_config.ProgressColumn(
                                    "Trophies", format="%d",
                                    min_value=0, max_value=max_trophies
                                ),
            "highest_trophies": st.column_config.NumberColumn("Highest 🏆",   format="%d"),
            "max_win_streak":   st.column_config.NumberColumn("🔥 Win Streak", format="%d"),
            "gadgets":          st.column_config.NumberColumn("Gadgets",      format="%d"),
            "star_powers":      st.column_config.NumberColumn("Star Powers",  format="%d"),
            "gears":            st.column_config.NumberColumn("Gears",        format="%d"),
        },
        hide_index=True
    )

    st.markdown("---")

    # ── Roster Summary ───────────────────────────────────────
    st.subheader("📊 Roster Summary")

    df_full = pd.DataFrame([
        {
            "power":       b.get("power", 0),
            "rank":        b.get("rank", 0),
            "gadgets":     len(b.get("gadgets", [])),
            "star_powers": len(b.get("starPowers", [])),
            "gears":       len(b.get("gears", [])),
        }
        for b in brawlers_api
    ])

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("⚡ Max Power (11)",     int((df_full["power"] == 11).sum()))
    s2.metric("🏅 Rank 35+",           int((df_full["rank"] >= 35).sum()))
    s3.metric("🔧 All Gadgets (2/2)",  int((df_full["gadgets"] == 2).sum()))
    s4.metric("⭐ All Star Powers",    int((df_full["star_powers"] == 2).sum()))
    s5.metric("⚙️ All Gears (2/2)",   int((df_full["gears"] == 2).sum()))

else:
    st.info("No brawler data available for this player.")
