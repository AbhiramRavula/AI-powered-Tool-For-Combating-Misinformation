from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Public fake-news model
detector = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sst2",
    return_all_scores=True
)

def init_db():
    conn = sqlite3.connect('misinformation.db')
    conn.execute('''
      CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_type TEXT, input TEXT,
        label TEXT, score REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
      ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json or {}
    text = data.get('text', '').strip()
    url  = data.get('url', '').strip()

    if url:
        try:
            resp = requests.get(url, timeout=5)
            soup = BeautifulSoup(resp.content, 'html.parser')
            content = soup.get_text(separator=' ', strip=True)[:512]
            analysis = detector(content)
        except Exception as e:
            return jsonify(error=f"URL fetch failed: {e}"), 400

    elif text:
        analysis = detector(text[:512])
    else:
        return jsonify(error="Provide text or URL"), 400

    # Take top-scoring label
    best = max(analysis[0], key=lambda x: x['score'])
    label, score = best['label'], best['score']

    # Store
    conn = sqlite3.connect('misinformation.db')
    conn.execute(
        "INSERT INTO analyses(input_type,input,label,score) VALUES (?,?,?,?)",
        ('url' if url else 'text', url or text, label, float(score))
    )
    conn.commit()
    conn.close()

    return jsonify(label=label, score=round(score,2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
