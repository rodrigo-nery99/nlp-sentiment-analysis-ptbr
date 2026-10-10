import os
import re
import spacy
import unicodedata
import pandas as pd
import zipfile

def setup_spacy():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("pt_core_news_sm")
        return spacy.load("pt_core_news_sm")

nlp = setup_spacy()

def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def carregar_dados(caminho_csv='data/b2w.csv', amostra=20000):
    df = pd.read_csv(caminho_csv)
    df = df.sample(n=amostra, random_state=42).reset_index(drop=True)
    df = df.rename(columns={'review_text': 'text', 'rating': 'sentiment'})

    def mapear_sentimento(nota):
        if nota > 3: return 'Positivo'
        elif nota <3: return 'Negativo'
        else: return 'Neutro'

    if df['sentiment'].dtype in ['int64', 'float64']:
        df['sentiment'] = df['sentiment'].apply(mapear_sentimento)

    df = df.dropna(subset=['text', 'sentiment'])
    df['clean_text'] = df['text'].apply(clean_text)

    # Filtrar textos curtos
    df['clean_token_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df = df[df['clean_token_count'] >= 5].copy()

    return df 









