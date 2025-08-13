import pandas as pd

def load_data(stories_xlsx: str, criteria_xlsx: str):
    stories = pd.read_excel(stories_xlsx, sheet_name="stories")
    criteria = pd.read_excel(criteria_xlsx, sheet_name="criteria")
    return stories, criteria

def preprocess_text(text: str) -> str:
    return " ".join(str(text).split()).strip()
