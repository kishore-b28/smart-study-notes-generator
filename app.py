from flask import Flask, request, jsonify, send_from_directory
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/summarize", methods=["POST"])
def summarize():

    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter some text."
        }), 400

    if not HF_TOKEN:
        return jsonify({
            "error": "Hugging Face API token is not configured."
        }), 500

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 80,
            "min_length": 20
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return jsonify({
            "error": "Hugging Face API error: " + response.text
        }), 500

    result = response.json()

    summary = result[0]["summary_text"]

    original_words = len(text.split())
    summary_words = len(summary.split())

    reduction = (
        ((original_words - summary_words)
        / original_words) * 100
    )

    return jsonify({
        "summary": summary,
        "original_words": original_words,
        "summary_words": summary_words,
        "reduction": round(reduction, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
