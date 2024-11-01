from flask import Blueprint, jsonify, request
from utils.twitter_api import fetch_twitter_data

twitter_bp = Blueprint("twitter", __name__)

@twitter_bp.route("/api/twitter/search", methods=["GET"])
def search_twitter():
    query = request.args.get("query")
    search_type = request.args.get("search_type", "Top")

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        data = fetch_twitter_data(query, search_type)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
