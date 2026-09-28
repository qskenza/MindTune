from pathlib import Path
import csv
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "experiment_log.csv"


def log_experiment(face_emotion, watch_emotion, final_emotion, mode, generation_time):
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "face_emotion",
                "watch_emotion",
                "final_emotion",
                "mode",
                "generation_time_seconds"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            face_emotion,
            watch_emotion,
            final_emotion,
            mode,
            round(generation_time, 2)
        ])