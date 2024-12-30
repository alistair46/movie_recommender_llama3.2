import pandas as pd
import faiss
import requests
import numpy as np
from tqdm import tqdm  # progress bar in CLI

# Load data
df = pd.read_csv('/home/Python_llms_projects/netflix_titles.csv')

# Function to create text format
def create_text_format(row):
    text_format = f"""
Type: {row['type']}
Title: {row['title']}
Director: {row['director']}
Cast: {row['cast']}
Released: {row['release_year']}
Genres: {row['listed_in']}
Description: {row['description']}
"""
    return text_format

# Add the 'text_format' column
df['text_format'] = df.apply(create_text_format, axis=1)

# FAISS index setup
dim = 4069  # Ensure this matches the embedding size
index = faiss.IndexFlatL2(dim)

# Prepare numpy array for embeddings
x = np.zeros((len(df['text_format']), dim), dtype='float32')

# Fetch embeddings and add to FAISS index
for i, representation in tqdm(enumerate(df['text_format']), total=len(df['text_format'])):
    try:
        res = requests.post(
            'http://localhost:11434/api/embeddings',
            json={'model': 'llama3.2', 'prompt': representation},
        )
        res.raise_for_status()  # Raise an error for HTTP issues
        embedding = res.json().get('embedding')
        if embedding and len(embedding) == dim:
            x[i] = np.array(embedding, dtype='float32')
        else:
            print(f"Invalid embedding size for row {i}")
    except Exception as e:
        print(f"Error processing row {i}: {e}")

# Add embeddings to FAISS index
index.add(x)
