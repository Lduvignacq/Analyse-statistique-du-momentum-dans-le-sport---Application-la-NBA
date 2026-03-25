import os
import sys

import zipfile
from tqdm import tqdm
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    playbyplayv3,
    teamgamelog,
)
import time
import pandas as pd
import matplotlib as plt

SLEEP = 0.1
MAX_RETRIES = 3
RETRY_WAIT = 60


def api_call_with_retry(func, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = func(*args, **kwargs)
            time.sleep(SLEEP)
            return result
        except Exception as e:
            print(f"    API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                print(f"    Retrying in {RETRY_WAIT} seconds...")
                time.sleep(RETRY_WAIT)
            else:
                print(f"    Max retries reached, skipping.")
                raise


def get_game_details_for_team_season(team_id, season):
    print(f"Fetching games for team_id {team_id} in season {season}...")

    team_games = api_call_with_retry(teamgamelog.TeamGameLog, team_id=team_id, season=season)
    df_games = team_games.get_data_frames()[0]

    game_details_list = []

    for index, game in df_games.iterrows():
        game_id = game['Game_ID']
        game_date = game['GAME_DATE']
        matchup = game['MATCHUP']
        wl = game['WL']
        print(f"  Processing game: {game_id} - {matchup}")

        try:
            box = api_call_with_retry(boxscoretraditionalv2.BoxScoreTraditionalV2, game_id=game_id)
            df_teams_box = box.get_data_frames()[1]

            team_row = df_teams_box[df_teams_box['TEAM_ID'] == team_id]
            opponent_row = df_teams_box[df_teams_box['TEAM_ID'] != team_id]

            team_score = team_row['PTS'].iloc[0] if not team_row.empty else None
            opponent_score = opponent_row['PTS'].iloc[0] if not opponent_row.empty else None

            score_differential = team_score - opponent_score if team_score is not None and opponent_score is not None else None

            game_details_list.append({
                'team_id': team_id,
                'game_id': game_id,
                'game_date': game_date,
                'matchup': matchup,
                'wl': wl,
                'team_score': team_score,
                'ennemy_score': opponent_score,
                'score_differential': score_differential
            })

        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            game_details_list.append({
                'team_id': team_id,
                'game_id': game_id,
                'game_date': game_date,
                'matchup': matchup,
                'wl': wl,
                'team_score': None,
                'ennemy_score': None,
                'score_differential': None
            })

    df_game_details = pd.DataFrame(game_details_list)
    return df_game_details


START_YEAR = 2000
END_YEAR = 2025

base_output_dir = "all_scorelines"

if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

print(f"Starting data collection for seasons {START_YEAR}-{END_YEAR}...")

all_teams = teams.get_teams()

for team in tqdm(all_teams, desc="Processing Teams"):
    team_id = team['id']
    team_name_for_folder = team['full_name'].replace(' ', '_').replace('.', '')
    team_output_dir = os.path.join(base_output_dir, team_name_for_folder)

    if not os.path.exists(team_output_dir):
        os.makedirs(team_output_dir)

    print(f"\nProcessing team: {team['full_name']} (ID: {team_id})")

    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc=f"  Seasons for {team['abbreviation']}", leave=False):
        season_str = f"{year}-{str(year+1)[2:]}"
        output_csv_path = os.path.join(team_output_dir, f"{team_name_for_folder}_{season_str}_game_details.csv")

        if os.path.exists(output_csv_path):
            print(f"    Skipping {team['abbreviation']} - {season_str}, file already exists.")
            continue

        try:
            df_game_details = get_game_details_for_team_season(
                team_id=team_id,
                season=season_str
            )
            df_game_details.to_csv(output_csv_path, index=False)
            print(f"    Saved {len(df_game_details)} game details for {team['abbreviation']} - {season_str} to {output_csv_path}")

        except Exception as e:
            print(f"    Error fetching data for {team['full_name']} in season {season_str}: {e}")
            sys.exit(1)


        time.sleep(SLEEP)

    zip_file_name = os.path.join(base_output_dir, f"{team_name_for_folder}.zip")
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(team_output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(team_output_dir))
                zipf.write(file_path, arcname)
    print(f"  Zipped data for {team['full_name']} to {zip_file_name}")

print("\nData collection and zipping process completed.")