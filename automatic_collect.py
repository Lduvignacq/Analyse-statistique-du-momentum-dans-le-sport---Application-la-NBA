import subprocess
import time

MAX_RESTARTS = 10000000

for attempt in range(1, MAX_RESTARTS + 1):
    print(f"\n=== Starting attempt {attempt}/{MAX_RESTARTS} ===")
    result = subprocess.run(["python", "collect_score_differentials_all_teams.py"])

    if result.returncode == 0:
        print("Completed successfully!")
        break
    else:
        print(f"Script crashed (exit code {result.returncode}).")
        if attempt < MAX_RESTARTS:
            print("Restarting in 10 seconds...")
            time.sleep(10)
        else:
            print("Max restarts reached, giving up.")