# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Initialize model
detector = pipeline("text-classification", model="roberta-base-finetuned-fakenews")

def init_db():
    conn = sqlite3.connect('misinformation.db')
    conn.execute('''
      CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY, input_type TEXT, input TEXT,
        score REAL, confidence REAL, explanation TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
      ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    if text:
        result = detector(text[:512])
        return jsonify(result)
    return jsonify({"error": "No text provided"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
