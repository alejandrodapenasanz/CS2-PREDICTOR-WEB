from flasgger import Swagger
from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    app.json.sort_keys = False

    Swagger(app)

    from routes.matches import matches_bp
    from routes.news import news_bp
    from routes.players import players_bp
    from routes.results import results_bp
    from routes.teams import teams_bp

    app.register_blueprint(teams_bp)
    app.register_blueprint(players_bp)
    app.register_blueprint(matches_bp)
    app.register_blueprint(news_bp)
    app.register_blueprint(results_bp)

    return app


flask_app = create_app()

if __name__ == "__main__":
    flask_app.run(debug=True, host="0.0.0.0", port=8000)
