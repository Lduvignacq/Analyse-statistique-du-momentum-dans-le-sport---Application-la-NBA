import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import re

SLEEP = 1

from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    playbyplayv3,
    teamgamelog,
)
from nba_api.stats.static import teams, players


def nba_request_with_retry(endpoint_cls, max_retries: int = 4, backoff: float = 3.0, **kwargs):
    """Appelle un endpoint nba_api avec retry exponentiel."""
    delay = backoff
    for attempt in range(1, max_retries + 1):
        try:
            return endpoint_cls(timeout=60, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"Tentative {attempt}/{max_retries} échouée ({type(e).__name__}). "
                  f"Nouvelle tentative dans {delay:.0f}s…")
            time.sleep(delay)
            delay *= 2
    print("Imports OK")

def get_team_id(name: str) -> dict:
    """"Récupère l'ID d'une équipe à partir de son nom complet ou de son surnom."""
    result = teams.find_teams_by_full_name(name)
    if not result:
        result = teams.find_teams_by_nickname(name)
    if not result:
        raise ValueError(f"Équipe '{name}' introuvable.")
    return result[0]

def clock_to_elapsed(period: int, clock_str: str) -> float:
    """Convertit (quart, 'PTxMy.zS') en secondes totales écoulées depuis le tip-off."""
    try:
        m = re.match(r'PT(\d+)M([\d.]+)S', str(clock_str))
        mins, secs = float(m.group(1)), float(m.group(2))
    except Exception:
        return np.nan
    time_left = mins * 60 + secs
    period_duration = 5 * 60 if period > 4 else 12 * 60
    elapsed_in_period = period_duration - time_left
    base = min(period - 1, 4) * 12 * 60 + max(0, period - 5) * 5 * 60
    return base + elapsed_in_period

def get_pbp(game_id: str)-> pd.DataFrame:

    """Récupère le DataFrame du play by play du match gam_id"""
    
    game_id_str = str(game_id).zfill(10)

    time.sleep(SLEEP)
    pbp = nba_request_with_retry(playbyplayv3.PlayByPlayV3,game_id=game_id_str)
    time.sleep(SLEEP)

    df_pbp = pbp.get_data_frames()[0]
    print(f"Nombre d'actions : {len(df_pbp)}")
    print(f"Colonnes disponibles : {list(df_pbp.columns)}")

    return df_pbp

def get_extended_pbp(game_id: str) -> pd.DataFrame:
    
    """Rajoute le temps, le score et la différence de pt à chaque instant"""

    df_pbp = get_pbp(game_id)

    df = df_pbp.copy()
    
    # Temps écoulé depuis le tip-off (en secondes puis en minutes)
    df["ELAPSED_SECONDS"] = df.apply(
        lambda r: clock_to_elapsed(r["period"], r["clock"]), axis=1
    )
    df["ELAPSED_MINUTES"] = df["ELAPSED_SECONDS"] / 60
    
    # Scores cumulatifs (V3 fournit scoreHome / scoreAway directement)
    df["SCORE_HOME"]    = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0)
    df["SCORE_VISITOR"] = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0)
    
    # Marge de score (point de vue de l'équipe domicile)
    df["SCOREMARGIN_FF"] = df["SCORE_HOME"] - df["SCORE_VISITOR"]

    return df 

def get_box(game_id : str):

    """Returns box info"""

    time.sleep(SLEEP)
    box = nba_request_with_retry(
        boxscoretraditionalv2.BoxScoreTraditionalV2,
        game_id=game_id,
    )
    time.sleep(SLEEP)

    return box

def get_games_played(team_id: int, season: str):

    """Returns liste des matchs joués par une équipe dans une saison donnée"""

    time.sleep(SLEEP)
    gamelog = nba_request_with_retry(
        teamgamelog.TeamGameLog,
        team_id=team_id,
        season=season,
    )
    time.sleep(SLEEP)
    df_games = gamelog.get_data_frames()[0]
    return df_games

def point_filter1(game_id: str) -> pd.DataFrame:
    
    """Retourne les actions de scoring d'une équipe dans un match donné."""
    
    df= get_extended_pbp(game_id)
    
    # On ne garde que les tirs réussis et les lancers-francs convertis
    # V3 : isFieldGoal == 1 + shotResult == 'Made'  OU  actionType == 'Free Throw' + description contient 'PTS'
    mask_fg  = (df["isFieldGoal"] == 1) & (df["shotResult"] == "Made")
    mask_ft  = (df["actionType"] == "Free Throw") & (df["description"].str.contains("PTS", na=False))
    df_score_events = df[mask_fg | mask_ft].copy()
    # Identifier qui marque : location == 'h' → home, sinon → visitor
    df_score_events["SCORER"] = df_score_events["location"].apply(
        lambda loc: "HOME" if loc == "h" else "VISITOR"
    )

    # Détection des runs : séquence de scoring consécutive par la même équipe
    df_score_events = df_score_events.reset_index(drop=True)
    df_score_events["RUN_ID"] = (
        df_score_events["SCORER"] != df_score_events["SCORER"].shift()
    ).cumsum()

    # Variation de score à chaque événement
    df_score_events["HOME_DELTA"]    = df_score_events["SCORE_HOME"].diff().fillna(0).clip(lower=0)
    df_score_events["VISITOR_DELTA"] = df_score_events["SCORE_VISITOR"].diff().fillna(0).clip(lower=0)
    
    # Agrégation par run
    runs = df_score_events.groupby(["RUN_ID", "SCORER"]).agg(
        pts_home    =("HOME_DELTA",     "sum"),
        pts_visitor =("VISITOR_DELTA",  "sum"),
        start_min   =("ELAPSED_MINUTES","first"),
        end_min     =("ELAPSED_MINUTES","last"),
        n_events    =("actionType",     "count"),
    ).reset_index()

    runs["pts"] = runs.apply(
        lambda r: r["pts_home"] if r["SCORER"] == "HOME" else r["pts_visitor"], axis=1
    )
    
    return runs

def fetch_pbp_for_games(game_ids: list, sleep: float = 1.0) -> pd.DataFrame:
    """Récupère et concatène le play-by-play enrichi pour une liste de Game IDs (V3)."""
    frames = []
    for i, gid in enumerate(game_ids):
        gid_str = str(gid).zfill(10)
        try:
            pbp_tmp = nba_request_with_retry(
                playbyplayv3.PlayByPlayV3, game_id=gid_str
            ).get_data_frames()[0]

            pbp_tmp["GAME_ID"] = gid
            pbp_tmp["ELAPSED_SECONDS"] = pbp_tmp.apply(
                lambda r: clock_to_elapsed(r["period"], r["clock"]), axis=1
            )
            pbp_tmp["ELAPSED_MINUTES"] = pbp_tmp["ELAPSED_SECONDS"] / 60
            pbp_tmp["SCORE_HOME"]    = pd.to_numeric(pbp_tmp["scoreHome"], errors="coerce").ffill().fillna(0)
            pbp_tmp["SCORE_VISITOR"] = pd.to_numeric(pbp_tmp["scoreAway"], errors="coerce").ffill().fillna(0)
            pbp_tmp["SCOREMARGIN_FF"] = pbp_tmp["SCORE_HOME"] - pbp_tmp["SCORE_VISITOR"]
            frames.append(pbp_tmp)
            print(f" [{i+1}/{len(game_ids)}] Game {gid_str} OK ({len(pbp_tmp)} actions)")
        except Exception as e:
            print(f" [{i+1}/{len(game_ids)}] Game {gid_str} abandonné : {e}")
        time.sleep(sleep)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def calculate_weighted_score(row) -> tuple:
    """Calculates a weighted score for a play-by-play event and identifies the scoring team."""
    weighted_score = 0
    weighted_scorer = None

    # Determine the scoring team
    if pd.notna(row['location']): 
        weighted_scorer = 'HOME' if row['location'] == 'h' else 'VISITOR'

    # Assign weighted scores based on event type
    if row['actionType'] == 'Made Shot':
        weighted_score = row['shotValue']
    elif row['actionType'] == 'Free Throw':
        if row['description'] and 'PTS' in row['description']:
            weighted_score = 1 
        else:
            weighted_score = 0
            weighted_scorer = None 
    elif row['actionType'] == 'Rebound':
        weighted_score = 0.5
    elif row['actionType'] == 'Steal':
        weighted_score = 1.0
    elif row['actionType'] == 'Block':
        weighted_score = 0.75
    
    if weighted_score == 0:
        weighted_scorer = None

    return weighted_score, weighted_scorer


