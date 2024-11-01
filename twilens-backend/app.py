from flask import Flask
from flask_cors import CORS
from config import Config
from routes.twitter_routes import twitter_bp
from routes.analysis_routes import analysis_bp

app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS for frontend-backend communication
CORS(app)

# Register Blueprints
app.register_blueprint(twitter_bp)
app.register_blueprint(analysis_bp)

if __name__ == "__main__":
    app.run(debug=True)
