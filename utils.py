import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import genextreme
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

def get_home_flag(gid, team_id):
    """Renvoie +1 si team_id était HOME, -1 si VISITOR (signe de SCOREMARGIN)."""
    row = df_games[df_games["Game_ID"] == gid]
    if row.empty:
        return 1
    matchup = row["MATCHUP"].values[0]
    return 1 if "vs." in matchup else -1


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

def point_filter2(game_id: str) -> pd.DataFrame:
    """Returns a DataFrame of scoring events with weighted scores for a given game."""
    df = get_extended_pbp(game_id)

    # Calculate weighted scores and identify scoring team
    df[['WEIGHTED_SCORE', 'WEIGHTED_SCORER']] = df.apply(calculate_weighted_score, axis=1, result_type='expand')

    df_weighted_score_events = df[df['WEIGHTED_SCORE'] > 0].copy()

    # Identify who scores: 'HOME' or 'VISITOR' based on WEIGHTED_SCORER
    df_weighted_score_events["SCORER"] = df_weighted_score_events["WEIGHTED_SCORER"]

    # Group events by game and then identify consecutive weighted scoring by the same team
    df_weighted_score_events = df_weighted_score_events.sort_values(by=['gameId', 'ELAPSED_SECONDS']).reset_index(drop=True)
    df_weighted_score_events["WEIGHTED_RUN_ID"] = (
        df_weighted_score_events.groupby('gameId')['SCORER'].bfill() !=
        df_weighted_score_events.groupby('gameId')['SCORER'].bfill().shift()
    ).cumsum()

    # Aggregate by weighted run
    weighted_runs = df_weighted_score_events.groupby(['gameId', 'WEIGHTED_RUN_ID', 'SCORER']).agg(
        weighted_pts=("WEIGHTED_SCORE", "sum"),
        start_min=("ELAPSED_MINUTES", "first"),
        end_min=("ELAPSED_MINUTES", "last"),
        n_events=("actionType", "count")
    ).reset_index()

    return weighted_runs

def analyze_games_for_extreme_runs(game_ids: list, extreme_threshold: int = 8, extreme_weighted_threshold: int = 10) -> pd.DataFrame:
    """
    Analyzes a list of NBA game IDs for extreme raw and weighted runs,
    calculating the probability of observing such runs for each game.

    Args:
        game_ids (list): A list of NBA game IDs (strings).
        extreme_threshold (int): The point threshold for defining an 'extreme' raw run.
        extreme_weighted_threshold (int): The point threshold for defining an 'extreme' weighted run.

    Returns:
        pd.DataFrame: A DataFrame summarizing the extreme run probabilities for each game.
    """
    results = []

    for game_id in game_ids:
        print(f"Analyzing game ID: {game_id}...")
        try:
            # Analyze raw runs
            runs = point_filter1(game_id)
            total_runs_count = len(runs)
            extreme_runs = runs[runs['pts'] >= extreme_threshold]
            extreme_runs_count = len(extreme_runs)
            prob_extreme_raw = extreme_runs_count / total_runs_count if total_runs_count > 0 else 0

            # Analyze weighted runs
            weighted_runs = point_filter2(game_id)
            total_weighted_runs_count = len(weighted_runs)
            extreme_weighted_runs = weighted_runs[weighted_runs['weighted_pts'] >= extreme_weighted_threshold]
            extreme_weighted_runs_count = len(extreme_weighted_runs)
            prob_extreme_weighted = extreme_weighted_runs_count / total_weighted_runs_count if total_weighted_runs_count > 0 else 0

            results.append({
                'gameId': game_id,
                'prob_extreme_raw_run': f"{prob_extreme_raw:.4f}",
                'num_extreme_raw_runs': extreme_runs_count,
                'total_raw_runs': total_runs_count,
                'raw_threshold': extreme_threshold,
                'raw_run_points_distribution': runs['pts'].tolist(),
                'prob_extreme_weighted_run': f"{prob_extreme_weighted:.4f}",
                'num_extreme_weighted_runs': extreme_weighted_runs_count,
                'total_weighted_runs': total_weighted_runs_count,
                'weighted_threshold': extreme_weighted_threshold,
                'weighted_run_points_distribution': weighted_runs['weighted_pts'].tolist()
            })
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            results.append({
                'gameId': game_id,
                'prob_extreme_raw_run': None,
                'num_extreme_raw_runs': None,
                'total_raw_runs': None,
                'raw_threshold': extreme_threshold,
                'raw_run_points_distribution': None,
                'prob_extreme_weighted_run': None,
                'num_extreme_weighted_runs': None,
                'total_weighted_runs': None,
                'weighted_threshold': extreme_weighted_threshold,
                'weighted_run_points_distribution': None,
                'error': str(e)
            })

    return pd.DataFrame(results)

def analysis_extreme_run(season : str, team_id: str):
    df_games = get_games_played(team_id,season)
    all_game_ids = df_games['Game_ID'].unique().tolist()
    full_analysis_summary = analyze_games_for_extreme_runs(all_game_ids)
    return full_analysis_summary


def fit_gev_distribution(data):
    """
    Fits the Generalized Extreme Value (GEV) distribution to the given data.

    Args:
        data (array-like): The dataset to fit the GEV distribution to.

    Returns:
        tuple: A tuple containing the shape (c), location (loc),
               and scale (scale) parameters of the fitted GEV distribution.
    """
    # Fit the GEV distribution to the data
    c, loc, scale = genextreme.fit(data)
    return c, loc, scale

