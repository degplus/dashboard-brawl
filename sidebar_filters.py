import streamlit as st
import json
import datetime

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
    
    # 1. Initialize Session State
    for key in FILTER_KEYS:
        if key not in st.session_state:
            st.session_state[key] = []

    if st.session_state.get("clear_filters", False):
        for key in FILTER_KEYS:
            st.session_state[key] = []
            # Esvazia a memória temporária do botão também
            if f"widget_{key}" in st.session_state:
                st.session_state[f"widget_{key}"] = []
        st.session_state["clear_filters"] = False

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

        # Helper to get dynamic options
        def get_filter_options(exclude_key: str) -> list:
            mask = (df_dim["battle_date"] >= date_start) & (df_dim["battle_date"] <= date_end)

            for key, col in COL_MAP.items():
                if key == exclude_key:
                    continue
                values = st.session_state.get(key, [])
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

        for i, (key, label, emoji) in enumerate(filters_config):
            options = get_filter_options(key)
            widget_key = f"widget_{key}"
            
            # Se o botão nasceu agora (ex: mudou de página), ele pega os dados do Cofre
            if widget_key not in st.session_state:
                mochila = st.session_state.get(key, [])
                st.session_state[widget_key] = [item for item in mochila if item in options]
                
            # A função (Callback) que salva no Cofre sempre que o usuário clica
            def save_filter(k, wk):
                st.session_state[k] = st.session_state[wk]

            # Desenhamos o filtro
            st.multiselect(
                f"{emoji} {label}",
                options=options,
                key=widget_key,
                on_change=save_filter,
                kwargs={"k": key, "wk": widget_key},
                placeholder=f"All {label.lower()}s..."
            )
            
            # H2H Mode usa o Cofre oficial (st.session_state[key])
            if key == "f_team":
                if len(st.session_state[key]) == 2:
                    is_h2h_mode = st.toggle(
                        "⚔️ H2H Mode (Only Direct Matches)",
                        value=False,
                        key="h2h_toggle_active",
                        help="If active, shows ONLY metrics from games where these two teams played against each other."
                    )
                else:
                    is_h2h_mode = False

            if i == 0:
                st.markdown("---")

        st.markdown("---")
       
        if any(st.session_state.get(k) for k in FILTER_KEYS):
            if st.button("🗑️ Clear All Filters", use_container_width=True):
                st.session_state["clear_filters"] = True
                st.rerun()

    # 3. Build Queries
    def build_where(use_datetime=False):
        date_col   = "DATE(battle_time)" if use_datetime else "battle_date"
        conds      = [f"{date_col} BETWEEN @d_start AND @d_end"]
        raw_params = [
            {"name": "d_start", "bq_type": "DATE", "value": str(date_start)},
            {"name": "d_end",   "bq_type": "DATE", "value": str(date_end)},
        ]

        if is_h2h_mode and len(st.session_state.f_team) == 2:
            team_a = st.session_state.f_team[0]
            team_b = st.session_state.f_team[1]
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
            values = st.session_state.get(key, [])
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
            values = st.session_state.get(key, [])
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