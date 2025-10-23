# src/run_scheduler.py
import schedule
import time
import traceback
from datetime import datetime
from run_best import run_dream_algo
from utils_logger import log_status

def safe_run():
    try:
        log_status("RUNNING", f"Dream Algo started at {datetime.now()}")
        run_dream_algo()
        log_status("COMPLETE", f"Cycle finished at {datetime.now()}")
    except Exception as e:
        log_status("ERROR", f"{datetime.now()} | {e}")
        traceback.print_exc()

def daily_schedule():
    # 4 AM → 8 PM loop
    schedule.every().day.at("04:00").do(safe_run)
    schedule.every(15).minutes.do(safe_run)  # auto-refresh loop
   

if __name__ == "__main__":
    log_status("START", "Dream Algo Scheduler initiated.")
    daily_schedule()
    while True:
        schedule.run_pending()
        time.sleep(10)
