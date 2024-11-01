import requests

COLAB_BASE_URL = "https://<colab-url>.ngrok.io"  # Replace with Colab public URL

def analyze_sentiment(text):
    response = requests.post(f"{COLAB_BASE_URL}/sentiment", json={"text": text})
    response.raise_for_status()
    return response.json()["sentiment"]

def summarize_text(text):
    response = requests.post(f"{COLAB_BASE_URL}/summarize", json={"text": text})
    response.raise_for_status()
    return response.json()["summary"]
