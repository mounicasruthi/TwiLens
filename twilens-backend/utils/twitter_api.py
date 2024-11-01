import requests
from config import Config

def fetch_twitter_data(query, search_type="Top"):
    url = "https://twitter-api45.p.rapidapi.com/search.php"
    headers = {
        "x-rapidapi-key": Config.RAPIDAPI_KEY,
        "x-rapidapi-host": Config.RAPIDAPI_HOST
    }
    params = {"query": query, "search_type": search_type}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()  # Raises error if request fails

    return response.json()
