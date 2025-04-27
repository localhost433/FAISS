# app/core/text_utils.py

def combining_text(row):
    neighborhood = str(row.get('neighborhood', ''))
    tags = str(row.get('tags', ''))
    short_description = str(row.get('short_description', ''))
    emojis = str(row.get('emojis', ''))
    return f"Neighborhood: {neighborhood}. Tags: {tags}. {short_description}. Emojis: {emojis}"

def concat_reviews(series):
    return ' '.join(series.astype(str))
