import subprocess
import time

MAX_RESTARTS = 10000000000

for attempt in range(1, MAX_RESTARTS + 1):
    print(f"\n=== Starting attempt {attempt}/{MAX_RESTARTS} ===", flush=True)
    result = subprocess.run(["python3", "collect_score_differentials_all_teams.py"])

    print(f"Exit code: {result.returncode}", flush=True)

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