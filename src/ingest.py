import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
def load_csv(file_name: str) -> pd.DataFrame:
    """
    Load a CSV file from the data directory.
    """
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"{file_name} not found in data directory.")
    
    df = pd.read_csv(file_path)
    return df

def load_regular_season_results():
    return load_csv("MRegularSeasonCompactResults.csv")

def load_tournament_results():
    return load_csv("MNCAATourneyCompactResults.csv")

if __name__ == "__main__":
    print("Ingestion module ready.")
