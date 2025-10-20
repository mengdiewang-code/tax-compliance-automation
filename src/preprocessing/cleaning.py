
import pandas as pd

TEXT_COLS = ["vendor","category","country","method","description","invoice_id"]

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
        df["dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    for c in ["amount","vat_claimed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df
