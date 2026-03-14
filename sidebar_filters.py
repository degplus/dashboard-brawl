import streamlit as st
import json
import datetime
import time
from datetime import timezone

# Cria o relógio do cache lendo o secrets
_tournament_mode = st.secrets.get("tournament_mode", False) if "tournament_mode" in st.secrets else False
_ttl = 600 if _tournament_mode else 3600

@st.cache_data(ttl=_ttl)
def get_sidebar_cache_time():
    return time.time()

# ============================================================
# CONSTANTS
# ============================================================
FILTER_KEYS = ["f_region", "f_type", "f_mode", "f_map", "f_team", "f_player", "f_brawler"]
COL_MAP = {
    "f_region":  "source_player_region",
    "f_type":    "type",
    "f_mode":    "mode",
    "f_map":     "map",
    "f_team":    "player_team",
    "f_player":  "player_tag",
    "f_brawler": "brawler_name"
}

# ============================================================
# MAIN FUNCTION (THE LEGO PIECE)
# ============================================================
def render_sidebar_filters(df_dim, player_names, all_player_names):
    
    # 1. Initialize the Vault (O Cofre Mestre)
    if "filter_vault" not in st.session_state:
        st.session_state.filter_vault = {k: [] for k in FILTER_KEYS}

    # Função de Limpeza Total (Callback do botão)
    def clear_all_filters():
        for k in FILTER_KEYS:
            st.session_state.filter_vault[k] = []
            st.session_state[k] = [] # Limpa o visual do botão
        st.session_state["h2h_toggle_active"] = False

    # 2. Draw Sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("🔍 Filters")

        min_date = df_dim["battle_date"].min()
        max_date = df_dim["battle_date"].max()

        dates = st.date_input(
            "📅 Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(dates, (tuple, list)) and len(dates) == 2:
            date_start, date_end = dates
        else:
            date_start = date_end = (dates[0] if isinstance(dates, (tuple, list)) else dates)

        st.markdown("---")

        show_only_active = st.toggle(
            "👤 Show only active roster",
            value=True,
            help="On: Players list shows only active roster members. Off: Shows all players."
        )

        full_squad_only = st.toggle(
            "👥 Full Squad Only",
            value=True,
            help="On: Shows only games where all 3 tracked players from the same team played together."
        )

        st.markdown("---")

        # Helper to get dynamic options (LENDO DO COFRE MESTRE)
        def get_filter_options(exclude_key: str) -> list:
            mask = (df_dim["battle_date"] >= date_start) & (df_dim["battle_date"] <= date_end)

            for key, col in COL_MAP.items():
                if key == exclude_key:
                    continue
                values = st.session_state.filter_vault.get(key, [])
                if not values:
                    continue
                if key == "f_player":
                    names_dict  = player_names if show_only_active else all_player_names
                    name_to_tag = {v: k for k, v in names_dict.items()}
                    tags = [name_to_tag.get(n) for n in values if name_to_tag.get(n)]
                    if not tags:
                        mask &= False
                    else:
                        mask &= df_dim[col].isin(tags)
                else:
                    mask &= df_dim[col].isin(values)

            col = COL_MAP[exclude_key]
            raw = sorted(df_dim[mask][col].dropna().unique().tolist())

            if exclude_key == "f_player":
                names_dict = player_names if show_only_active else all_player_names
                return sorted([names_dict[tag] for tag in raw if tag in names_dict])

            return raw

        filters_config = [
            ("f_region",  "Region",  "🌍"),
            ("f_type",    "Type",    "📌"),
            ("f_mode",    "Mode",    "🎮"),
            ("f_map",     "Map",     "🗺️"),
            ("f_team",    "Team",    "🏆"),
            ("f_player",  "Player",  "👤"),
            ("f_brawler", "Brawler", "👊"),
        ]

        is_h2h_mode = False

        # Callback exigido pelo Streamlit para gravar a mudança instantaneamente
        def update_vault(k):
            st.session_state.filter_vault[k] = st.session_state[k]

        for i, (key, label, emoji) in enumerate(filters_config):
            options = get_filter_options(key)

            # CASCATA: Limpa opções antigas que não servem mais
            valid_selections = [x for x in st.session_state.filter_vault.get(key, []) if x in options]
            st.session_state.filter_vault[key] = valid_selections
            
            # Sincroniza o visual do botão FORÇADAMENTE com as opções válidas
            st.session_state[key] = valid_selections

            st.multiselect(
                f"{emoji} {label}",
                options=options,
                default=valid_selections,
                key=key,
                on_change=update_vault,
                kwargs={"k": key},
                placeholder=f"All {label.lower()}s..."
            )
            
            if key == "f_team":
                if len(st.session_state.filter_vault[key]) == 2:
                    is_h2h_mode = st.toggle(
                        "⚔️ H2H Mode (Only Direct Matches)",
                        value=st.session_state.get("h2h_toggle_active", False),
                        key="h2h_toggle_active",
                        help="If active, shows ONLY metrics from games where these two teams played against each other."
                    )
                else:
                    is_h2h_mode = False

            if i == 0:
                st.markdown("---")

        st.markdown("---")
       
        if any(st.session_state.filter_vault[k] for k in FILTER_KEYS):
            st.button("🗑️ Clear All Filters", use_container_width=True, on_click=clear_all_filters)

        # --- CACHE & REFRESH INFO ---
        st.markdown("---")
        cache_ts = get_sidebar_cache_time()
        loaded_at = datetime.datetime.fromtimestamp(cache_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        mins_ago = int((time.time() - cache_ts) / 60)
        ago_text = "just now" if mins_ago == 0 else "1 min ago" if mins_ago == 1 else f"{mins_ago} min ago"
        
        ttl_mins = _ttl // 60
        mode_label = "🏆 Tournament Mode" if _tournament_mode else "⚙️ Normal Mode"
        
        st.success(f"⚡ Cache loaded at {loaded_at}")
        st.caption(f"🕐 Updated {ago_text}")
        st.caption(f"🔄 Refreshing every {ttl_mins} min  |  {mode_label}")

    # 3. Build Queries (USANDO O COFRE)
    def build_where(use_datetime=False):
        date_col   = "DATE(battle_time)" if use_datetime else "battle_date"
        conds      = [f"{date_col} BETWEEN @d_start AND @d_end"]
        raw_params = [
            {"name": "d_start", "bq_type": "DATE", "value": str(date_start)},
            {"name": "d_end",   "bq_type": "DATE", "value": str(date_end)},
        ]

        if is_h2h_mode and len(st.session_state.filter_vault["f_team"]) == 2:
            team_a = st.session_state.filter_vault["f_team"][0]
            team_b = st.session_state.filter_vault["f_team"][1]
            conds.append(f"""
                game IN (
                    SELECT game FROM `brawl-sandbox.brawl_stats.vw_battles_python`
                    WHERE player_team IN (@h2h_t1, @h2h_t2)
                    GROUP BY game HAVING COUNT(DISTINCT player_team) = 2
                )
            """)
            raw_params.append({"name": "h2h_t1", "bq_type": "STRING", "value": team_a})
            raw_params.append({"name": "h2h_t2", "bq_type": "STRING", "value": team_b})

        if full_squad_only:
            conds.append("""
                game IN (
                    SELECT game FROM `brawl-sandbox.brawl_stats.vw_battles_python`
                    WHERE player_team IS NOT NULL
                    GROUP BY game, player_team HAVING COUNT(DISTINCT player_tag) >= 3
                )
            """)

        for key, col in COL_MAP.items():
            values = st.session_state.filter_vault.get(key, [])
            if not values:
                continue
            if is_h2h_mode and key == "f_team":
                continue
            if key == "f_player":
                names_dict  = player_names if show_only_active else all_player_names
                name_to_tag = {v: k for k, v in names_dict.items()}
                tags = [name_to_tag.get(n) for n in values if name_to_tag.get(n)]
                if tags:
                    conds.append(f"{col} IN UNNEST(@{key})")
                    raw_params.append({"name": key, "bq_type": "STRING", "value": tags})
            else:
                conds.append(f"{col} IN UNNEST(@{key})")
                raw_params.append({"name": key, "bq_type": "STRING", "value": values})

        return " AND ".join(conds), json.dumps(raw_params)

    def build_where_h2h():
        conds      = ["DATE(battle_time) BETWEEN @d_start AND @d_end"]
        raw_params = [
            {"name": "d_start", "bq_type": "DATE", "value": str(date_start)},
            {"name": "d_end",   "bq_type": "DATE", "value": str(date_end)},
        ]

        for key, col in COL_MAP.items():
            if key == "f_team":
                continue
            values = st.session_state.filter_vault.get(key, [])
            if not values:
                continue
            if key == "f_player":
                names_dict  = player_names if show_only_active else all_player_names
                name_to_tag = {v: k for k, v in names_dict.items()}
                tags = [name_to_tag.get(n) for n in values if name_to_tag.get(n)]
                if tags:
                    conds.append(f"{col} IN UNNEST(@{key})")
                    raw_params.append({"name": key, "bq_type": "STRING", "value": tags})
            else:
                conds.append(f"{col} IN UNNEST(@{key})")
                raw_params.append({"name": key, "bq_type": "STRING", "value": values})

        return " AND ".join(conds), raw_params

    where_main, params_main = build_where(use_datetime=True)
    where_h2h, base_params_h2h = build_where_h2h()

    # Retorna todos os "fios" para plugar na página principal
    return {
        "where_main": where_main,
        "params_main": params_main,
        "where_h2h": where_h2h,
        "base_params_h2h": base_params_h2h,
        "show_only_active": show_only_active,
        "full_squad_only": full_squad_only
    }