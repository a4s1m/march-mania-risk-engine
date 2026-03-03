from src.ingest import load_csv

if __name__ == "__main__":
    df = load_csv("MTeams.csv")
    print("Shape:", df.shape)
    print(df.head())


