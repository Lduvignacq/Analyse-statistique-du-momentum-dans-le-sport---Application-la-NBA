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

    # Calculate ELAPSED_MINUTES of the previous action in the full pbp
    # For the very first event, previous elapsed minutes will be 0.
    df['PREVIOUS_ELAPSED_MINUTES'] = df['ELAPSED_MINUTES'].shift(1).fillna(0)

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
    # start_min is now the ELAPSED_MINUTES of the action immediately preceding the first scoring action of the run.
    # end_min is the ELAPSED_MINUTES of the last scoring action of the run.
    runs = df_score_events.groupby(["RUN_ID", "SCORER"]).agg(
        pts_home    =("HOME_DELTA",     "sum"),
        pts_visitor =("VISITOR_DELTA",  "sum"),
        start_min   =("PREVIOUS_ELAPSED_MINUTES","first"),
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

def get_home_flag(gid: str, team_id: int, df_games_for_team: pd.DataFrame) -> bool:
    """
    Renvoie True si team_id était l'équipe à domicile pour le game_id donné,
    False si c'était l'équipe visiteuse.
    df_games_for_team doit être le gamelog de l'équipe 'team_id' pour la saison entière.
    """
    row = df_games_for_team[df_games_for_team["Game_ID"] == gid]
    if row.empty:
        raise ValueError(f"Game ID {gid} not found in the provided season game log for team {team_id}.")
    
    matchup = row["MATCHUP"].values[0]
    # 'MATCHUP' column for a team's gamelog indicates 'vs.' for home games and '@' for away games.
    return "vs." in matchup


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

    # Calculate ELAPSED_MINUTES of the previous action in the full pbp
    # For the very first event, previous elapsed minutes will be 0.
    df['PREVIOUS_ELAPSED_MINUTES'] = df['ELAPSED_MINUTES'].shift(1).fillna(0)

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
    # start_min is now the ELAPSED_MINUTES of the action immediately preceding the first scoring action of the run.
    # end_min is the ELAPSED_MINUTES of the last scoring action of the run.
    weighted_runs = df_weighted_score_events.groupby(['gameId', 'WEIGHTED_RUN_ID', 'SCORER']).agg(
        weighted_pts=("WEIGHTED_SCORE", "sum"),
        start_min=("PREVIOUS_ELAPSED_MINUTES", "first"),
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

def get_blowout_games_and_scores(team_id, season, percentile_threshold=90):
    
    print(f"Analyzing season {season} for team ID {team_id} with percentile threshold {percentile_threshold}")

    df_games = get_games_played(team_id, season)
    if df_games.empty:
        print("No games found for the specified team and season.")
        return []

    game_score_diffs = []

    for index, game_row in df_games.iterrows():
        game_id = game_row['Game_ID']
        try:
            box = get_box(game_id=game_id)
            df_teams_box = box.get_data_frames()[1]

            team_pts = df_teams_box[df_teams_box['TEAM_ID'] == team_id]['PTS'].iloc[0]
            opponent_pts = df_teams_box[df_teams_box['TEAM_ID'] != team_id]['PTS'].iloc[0]

            score_difference = team_pts - opponent_pts
            game_score_diffs.append((game_id, score_difference))
        except Exception as e:
            print(f"Could not retrieve box score for game {game_id}: {e}")
            continue

    if not game_score_diffs:
        print("No score differences could be calculated.")
        return []

    # Separate positive and negative score differences for percentile calculation
    all_score_differences = [diff for _, diff in game_score_diffs]
    positive_score_diffs = [diff for diff in all_score_differences if diff > 0]
    negative_score_diffs = [diff for diff in all_score_differences if diff < 0]

    winning_blowout_threshold = 0
    if positive_score_diffs:
        winning_blowout_threshold = np.percentile(positive_score_diffs, percentile_threshold)
    else:
        print("No positive score differences to calculate winning blowout threshold.")

    losing_blowout_threshold = 0
    if negative_score_diffs:
        losing_blowout_threshold = np.percentile(negative_score_diffs, 100 - percentile_threshold)
    else:
        print("No negative score differences to calculate losing blowout threshold.")

    blowout_games = []
    for game_id, score_diff in game_score_diffs:
        if score_diff >= winning_blowout_threshold or score_diff <= losing_blowout_threshold:
            blowout_games.append((game_id, score_diff))

    print(f"Identified {len(blowout_games)} blowout games for {season}.")
    return blowout_games


