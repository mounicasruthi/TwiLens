from flask import Blueprint, jsonify, request
from utils.colab_client import analyze_sentiment, summarize_text

analysis_bp = Blueprint("analysis", __name__)

@analysis_bp.route("/api/analysis/sentiment", methods=["POST"])
def sentiment_analysis():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "Text parameter is required"}), 400

    try:
        sentiment = analyze_sentiment(text)
        return jsonify({"sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@analysis_bp.route("/api/analysis/summarize", methods=["POST"])
def summarize():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "Text parameter is required"}), 400

    try:
        summary = summarize_text(text)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
