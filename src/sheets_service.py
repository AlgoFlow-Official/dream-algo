import gspread
from google.oauth2.service_account import Credentials

def get_sheets_client():
    """
    Authenticate and return a gspread client for Google Sheets access.
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_file("my-algo-sheets-dbfb0c9f3187.json", scopes=scopes)
    client = gspread.authorize(creds)
    return client

def export_to_existing_sheet(dataframe):
    """
    Updates your existing Google Sheet with new DataFrame data.
    """
    sheet_id = "1RG3FApl9WUQB9efeA_UZAsLyDnieK1B9siT1-cEP4qk"
    client = get_sheets_client()
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.sheet1

    # Clear old data first
    worksheet.clear()

    # Convert DataFrame to list of lists (headers + rows)
    data = [dataframe.columns.values.tolist()] + dataframe.values.tolist()
    worksheet.update(data)

    print(f"[INFO] Updated your Google Sheet successfully!")
    print(f"🔗 https://docs.google.com/spreadsheets/d/{sheet_id}")
