# src/utils_logger.py
from datetime import datetime
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "dream_algo_log.txt")

def log_status(status, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{status}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
