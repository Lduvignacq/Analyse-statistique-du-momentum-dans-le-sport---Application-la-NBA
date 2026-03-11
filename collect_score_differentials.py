"""
collect_score_differentials.py
-------------------------------
Collecte les différentiels de score FINAL (du point de vue de chaque équipe)
pour toutes les équipes NBA sur une saison donnée, et sauvegarde le résultat
dans un CSV : data/score_differentials.csv

Usage :
    python collect_score_differentials.py

Colonnes du CSV :
    team_id, team_name, game_id, game_date, matchup, wl, score_differential
"""

import time
import pandas as pd
from pathlib import Path

from nba_api.stats.endpoints import teamgamelog, playbyplayv3
from nba_api.stats.static import teams

# ================================================================
# PARAMÈTRES
# ================================================================
SEASON    = "2024-25"
SLEEP     = 1.2          # délai entre requêtes (rate limiting NBA API)
OUTPUT    = Path("data/score_differentials.csv")

# ================================================================
# FONCTIONS UTILITAIRES
# ================================================================
def nba_request_with_retry(endpoint_cls, max_retries=4, backoff=3.0, **kwargs):
    """Appelle un endpoint nba_api avec retry exponentiel."""
    delay = backoff
    for attempt in range(1, max_retries + 1):
        try:
            return endpoint_cls(timeout=60, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"    ⚠️  Tentative {attempt}/{max_retries} échouée ({type(e).__name__}). "
                  f"Retry dans {delay:.0f}s…")
            time.sleep(delay)
            delay *= 2


def fetch_team_differentials(team_id: int, team_name: str) -> list[dict]:
    """
    Récupère tous les matchs d'une équipe sur la saison et calcule
    le différentiel de score final du point de vue de cette équipe.
    """
    # 1. Game log de la saison
    try:
        gl = nba_request_with_retry(
            teamgamelog.TeamGameLog,
            team_id=team_id,
            season=SEASON,
            season_type_all_star="Regular Season",
        )
        time.sleep(SLEEP)
    except Exception as e:
        print(f"  ❌ Impossible de récupérer le game log de {team_name} : {e}")
        return []

    df_games = gl.get_data_frames()[0]
    records  = []

    # 2. Pour chaque match, récupérer le play-by-play et le différentiel final
    for i, row in df_games.iterrows():
        gid     = row["Game_ID"]
        gid_str = str(gid).zfill(10)
        matchup = row["MATCHUP"]
        sign    = 1 if "vs." in matchup else -1   # HOME = +1, VISITOR = -1

        try:
            pbp = nba_request_with_retry(
                playbyplayv3.PlayByPlayV3, game_id=gid_str
            ).get_data_frames()[0]

            score_home    = pd.to_numeric(pbp["scoreHome"], errors="coerce").ffill().fillna(0)
            score_visitor = pd.to_numeric(pbp["scoreAway"], errors="coerce").ffill().fillna(0)
            margin        = (score_home - score_visitor).iloc[-1]
            diff          = margin * sign

            records.append({
                "team_id"           : team_id,
                "team_name"         : team_name,
                "game_id"           : gid,
                "game_date"         : row["GAME_DATE"],
                "matchup"           : matchup,
                "wl"                : row["WL"],
                "score_differential": diff,
            })
            print(f"    ✅ {matchup} ({row['GAME_DATE']})  →  diff={diff:+.0f}")

        except Exception as e:
            print(f"    ❌ Game {gid_str} abandonné : {e}")

        time.sleep(SLEEP)

    return records


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    all_teams = teams.get_teams()
    print(f"▶ {len(all_teams)} équipes NBA — Saison {SEASON}\n")

    all_records = []

    for idx, team in enumerate(all_teams):
        team_id   = team["id"]
        team_name = team["full_name"]
        print(f"[{idx+1:02d}/{len(all_teams)}] {team_name}")

        records = fetch_team_differentials(team_id, team_name)
        all_records.extend(records)

        # Sauvegarde intermédiaire après chaque équipe (sécurité)
        pd.DataFrame(all_records).to_csv(OUTPUT, index=False)
        print(f"  → {len(records)} matchs collectés | Total : {len(all_records)} | Sauvegardé dans {OUTPUT}\n")

    df_final = pd.DataFrame(all_records)
    df_final.to_csv(OUTPUT, index=False)
    print(f"\n✅ Terminé ! {len(df_final)} lignes sauvegardées dans {OUTPUT}")
    print(df_final.head(10).to_string())
