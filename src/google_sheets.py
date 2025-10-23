"""
google_sheets.py
-------------------------------------
Handles sending trade data from Dream Algo → Google Sheets.
Requires your Google service account JSON (my-algo-sheets-xxxx.json)
and gspread + pandas installed.
"""

import time
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials

# ==============================================
# 🔧 CONFIGURATION
# ==============================================
CREDENTIALS_FILE = "my-algo-sheets-dbfb0c9f3187.json"   # your credentials file
SPREADSHEET_ID = "1AN8dDvA3VkVJOZMmxd4-ZJKarFLksir5jIUVtBfVgKg"  # your sheet ID
SHEET_NAME = "Sheet1"  # rename if needed

# ==============================================
# 🚀 MAIN FUNCTION
# ==============================================
def update_google_sheet(rows):
    """
    Append trade rows to the configured Google Sheet.
    Expected format for each row dict:
        {
            "ticker": "AAPL",
            "price": 187.32,
            "change_pct": 0.8,
            "volume": 5432100,
            "signal": "buy",
            "confidence": 92,
            "ai_summary": "Apple stock up on iPhone 17 news"
        }
    """
    try:
        # --- Authorize client ---
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

        # --- Build final rows (include timestamp as first column) ---
        final_rows = []
        for row in rows:
            final_rows.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp first
                row.get("ticker", ""),
                row.get("price", ""),
                row.get("change_pct", ""),
                row.get("volume", ""),
                row.get("signal", ""),
                row.get("confidence", ""),
                row.get("ai_summary", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # last updated
            ])

        # --- Create headers if sheet is empty ---
        existing = sheet.get_all_values()
        if not existing:
            headers = [
                "TIMESTAMP",
                "TICKER",
                "PRICE",
                "CHANGE %",
                "VOLUME",
                "SIGNAL",
                "CONFIDENCE",
                "AI_SUMMARY",
                "LAST_UPDATED"
            ]
            sheet.append_row(headers)
            time.sleep(1)

        # --- Append new data ---
        sheet.append_rows(final_rows, value_input_option="USER_ENTERED")
        print(f"[SHEETS] ✅ {len(final_rows)} rows added successfully.")

    except Exception as e:
        print(f"[SHEETS] ❌ Error updating Google Sheet: {e}")
        time.sleep(3)
        try:
            # Retry once
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
            client = gspread.authorize(creds)
            sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
            sheet.append_rows(final_rows, value_input_option="USER_ENTERED")
            print("[SHEETS] 🔁 Retry succeeded.")
        except Exception as e2:
            print(f"[SHEETS] ❌ Retry failed: {e2}")
