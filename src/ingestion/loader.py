import pandas as pd


def load_netflix_data(file_path="data/netflix_titles.csv"):
    df = pd.read_csv(file_path)
    return df