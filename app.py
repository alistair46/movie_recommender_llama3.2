from flask import Flask, request, jsonify, render_template
import pandas as pd
import faiss
import requests
import numpy as np
from pathlib import Path


app = Flask(__name__)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent
#print(BASE_DIR)


# Load data and FAISS index
df = pd.read_csv(BASE_DIR / 'netflix_titles.csv')
faiss_index = faiss.read_index(str(BASE_DIR / 'faiss_index.bin'))


# Helper function
def create_text_format(row):
    """Formats a single row into a descriptive text."""
    text_format = f"""
    Type: {row.get('type', 'Unknown')}
    Title: {row.get('title', 'Unknown')}
    Director: {row.get('director', 'Unknown')}
    Cast: {row.get('cast', 'Unknown')}
    Released: {row.get('release_year', 'Unknown')}
    Genres: {row.get('listed_in', 'Unknown')}
    Description: {row.get('description', 'No description available')}
    """
    return text_format.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_prompt = data.get('prompt', '')

    # Get embedding from external service
    try:
        res = requests.post('http://localhost:11434/api/embeddings', json={'model': 'llama2', 'prompt': user_prompt})
        res.raise_for_status()
        embedding = np.array([res.json()['embedding']], dtype='float32')
    except Exception as e:
        return jsonify({"error": f"Failed to fetch embeddings: {str(e)}"}), 500

    # Search FAISS index
    D, I = faiss_index.search(embedding, 5)

    # Retrieve and format results
    recommendations = []
    seen_titles = set()
    for idx in I.flatten():
        if idx < len(df):
            match_row = df.iloc[idx]
            title = match_row.get('title', 'Unknown')
            if title not in seen_titles:
                seen_titles.add(title)
                recommendations.append(create_text_format(match_row))

    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
