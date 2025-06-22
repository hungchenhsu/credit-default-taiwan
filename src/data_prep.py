import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "default_of_credit_card_clients.xls"

def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df.set_index("ID", inplace=True)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}, inplace=True)
    df["MARRIAGE"].replace({0: 3}, inplace=True)
    return df