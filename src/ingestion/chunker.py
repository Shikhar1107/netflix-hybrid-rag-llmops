from langchain_core.documents import Document
import pandas as pd

def safe_value(value):
    if pd.isna(value):
        return "Unknown"
    return str(value).strip()

def create_documents(df):
    docs = []
    for _,row in df.iterrows():
        content = f"""
        Title: {safe_value(row['title'])},
        Type: {safe_value(row['type'])},
        Director: {safe_value(row['director'])},
        Cast: {safe_value(row['cast'])},
        Genre: {safe_value(row['listed_in'])},
        Country: {safe_value(row['country'])},
        Release Year: {safe_value(row['release_year'])},
        Rating: {safe_value(row['rating'])},
        Duration: {safe_value(row['duration'])},
        Description: {safe_value(row['description'])},
        """.strip()

        docs.append(Document(
            page_content=content,
            metadata={
                'title': safe_value(row['title']),
                'type': safe_value(row['type']),
                'genre': safe_value(row['listed_in']),
                'director': safe_value(row['director']),
                'cast': safe_value(row['cast']),
                'year': safe_value(row['release_year']),
                'rating': safe_value(row['rating']),
                'duration': safe_value(row['duration']),
                "country": safe_value(row["country"]),
            }
        ))
    
    return docs