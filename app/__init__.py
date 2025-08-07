import logging
import os

from flask import Flask

from app.config import Config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode="a"),
    ],
)


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Registrar blueprints
    from app.api.routes import api_bp

    app.register_blueprint(api_bp)

    # Crear directorios si no existen
    for folder in [app.config["UPLOAD_FOLDER"], "models", "results"]:
        os.makedirs(folder, exist_ok=True)

    return app
