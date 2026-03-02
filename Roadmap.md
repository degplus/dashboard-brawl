# 📊 DegStats — Brawl Stars Competitive Analytics

> Dashboard de análise estatística de partidas competitivas de Brawl Stars,
> construído com Streamlit + Google BigQuery.

---

## 🗂️ Estrutura do Projeto

```
~/dashboard-brawl/
├── assets/
│   └── logo.jpg                  # Logo do app (sidebar + ícone)
├── pages/
│   ├── 1_Player_Scout.py         # Página: Scout de Jogadores
│   └── 2_Player_Profile-2.py     # Página: Perfil do Jogador
├── venv/                         # Ambiente virtual Python
│   └── pyvenv.cfg
├── Overview.py                   # Página principal (entry point)
└── requirements.txt              # Dependências do projeto
```

---

## 🚀 Como Executar

### 1. Ativar o ambiente virtual
```bash
cd ~/dashboard-brawl
source venv/bin/activate
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Configurar secrets do Streamlit
Crie o arquivo `~/.streamlit/secrets.toml` com:
```toml
gcp_project = "brawl-sandbox"

[gcp_service_account]
type = "service_account"
project_id = "brawl-sandbox"
private_key_id = "..."
private_key = "..."
client_email = "..."
# ... demais campos da service account
```

### 4. Rodar o app
```bash
streamlit run Overview.py
```

---

## 🧰 Stack Tecnológica

| Tecnologia | Versão | Uso |
|---|---|---|
| Streamlit | >=1.30.0 | Framework do dashboard |
| Google BigQuery | 3.11.4 | Fonte de dados principal |
| Google Auth | 2.23.0 | Autenticação GCP |
| Pandas | >=2.0.0 | Manipulação de dados |
| Plotly | >=5.18.0 | Gráficos interativos |
| Pillow | >=10.0.0 | Processamento de imagens |
| Requests | 2.31.0 | Download de imagens via URL |
| db-dtypes | >=1.1.0 | Tipos de dados BigQuery |
| openpyxl | latest | Exportação Excel |
| streamlit-authenticator | >=0.3.3 | Sistema de login (legado) |

---

## 🗄️ Fonte de Dados — BigQuery

- **Projeto GCP:** `brawl-sandbox`
- **Dataset:** `brawl_stats`

### Tabelas e Views Utilizadas

| Objeto | Tipo | Descrição |
|---|---|---|
| `dim_filters` | Tabela | Dimensão de filtros (datas, regiões, modos, mapas, times, brawlers) |
| `vw_battles_python` | View | View principal de partidas com todos os campos de batalha |
| `dim_source_players` | Tabela | Cadastro de jogadores monitorados (`is_active`, `PL_TAG`) |

### Principais campos de `vw_battles_python`
- `game` — ID único da partida
- `battle_time` — Timestamp da batalha
- `player_tag` / `player_name` — Identificação do jogador
- `player_team` — Time do jogador
- `player_result` — `victory` ou `defeat`
- `brawler_name` / `brawler_img` — Brawler usado
- `team_num` / `player_place` — Posição no draft
- `star_player_name` — MVP da partida
- `mode` / `map` / `type` — Contexto da partida

---

## 📄 Páginas do App

### 🏠 Overview (`Overview.py`)
Página principal e entry point do Streamlit.

**Funcionalidades:**
- Visão geral de métricas competitivas
- Sistema de filtros globais (afeta todas as queries)
- Visualizador de Draft por partida
- Toggle H2H Mode para confrontos diretos entre times

**Sistema de Cache:**
- TTL de **3.600 segundos (1 hora)** com persistência em disco
- Auto-refresh automático ao expirar
- Carregamento paralelo de imagens via `ThreadPoolExecutor` (10 workers)

---

### 🔍 Player Scout (`pages/1_Player_Scout.py`)
Análise de desempenho individual de jogadores para fins de recrutamento.

**Funcionalidades:**
- Estatísticas detalhadas por jogador
- Performance por brawler
- Comparação entre jogadores
- Métricas de win rate e consistência

---

### 👤 Player Profile (`pages/2_Player_Profile-2.py`)
Perfil completo de um jogador específico.

**Funcionalidades:**
- Visão consolidada do histórico do jogador
- Brawlers mais utilizados
- Evolução de desempenho ao longo do tempo

---

## 🔍 Sistema de Filtros

Os filtros ficam na **sidebar** e são globais (afetam todas as páginas via `st.session_state`).

| Filtro | Chave | Coluna BQ |
|---|---|---|
| Região | `f_region` | `source_player_region` |
| Tipo | `f_type` | `type` |
| Modo | `f_mode` | `mode` |
| Mapa | `f_map` | `map` |
| Time | `f_team` | `player_team` |
| Jogador | `f_player` | `player_tag` |
| Brawler | `f_brawler` | `brawler_name` |

**Toggle — Roster Ativo:**
- `ON` → lista apenas jogadores com `is_active = TRUE` em `dim_source_players`
- `OFF` → lista todos os jogadores encontrados nas partidas

---

## ⚔️ H2H Mode (Head-to-Head)

Ativado automaticamente quando **exatamente 2 times** são selecionados no filtro de times.

**Lógica:**
- Filtra apenas partidas onde **ambos os times jogaram entre si** na mesma partida
- Usa subquery no BigQuery agrupando por `game` e verificando `COUNT(DISTINCT player_team) = 2`
- O filtro padrão de time é substituído pelo filtro H2H especial

---

## 🃏 Draft Viewer

Função `render_draft(game_id, map_name, map_img, bt)` que exibe:
- Times **Azul** e **Vermelho** com seus brawlers
- Resultado da partida (🏆 vitória / 💀 derrota)
- ⭐ Destaque do Star Player (MVP)
- Imagem do mapa

---

## 🖼️ Carregamento de Imagens

Imagens de brawlers são carregadas via URL externa e convertidas para **base64** para evitar problemas de CORS e melhorar a performance:

```python
@st.cache_data(ttl=86400, persist="disk")
def img_to_base64(url: str) -> str | None:
    ...  # Download + encode base64
```

- Cache de **24 horas** para imagens
- Download paralelo com `ThreadPoolExecutor` (10 workers)

---

## 🔐 Autenticação (Legado)

O sistema de login com `streamlit-authenticator` está **comentado** no código atual (`Overview.py`), permitindo acesso livre ao dashboard. O código está preservado para reativação futura se necessário.

---

## 📦 Atualização de Dependências

```bash
pip install -r requirements.txt --upgrade
```

---

*DegStats — Desenvolvido para análise competitiva de Brawl Stars* 🎮
