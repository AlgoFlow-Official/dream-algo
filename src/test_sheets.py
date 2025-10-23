import pandas as pd
from sheets_service import export_to_existing_sheet

# Dummy test data
df = pd.DataFrame({
    "Ticker": ["AAPL", "TSLA", "NVDA"],
    "Predicted Gain %": [4.5, 6.2, 3.8],
    "Date": ["2025-10-08", "2025-10-08", "2025-10-08"]
})

export_to_existing_sheet(df)
print("✅ Google Sheet updated successfully!")
from datetime import datetime
from google_sheets import update_google_sheet

# Create one sample trade row
rows = [
    {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": "TEST",
        "price": 100.25,
        "change_pct": 1.23,
        "volume": 123456,
        "signal": "buy",
        "confidence": 95
    }
]

# Try pushing to Google Sheets
print("[TEST] Sending sample row to Google Sheets...")
update_google_sheet(rows)
print("[TEST] Done ✅")
