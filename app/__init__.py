import logging
import logging.config
import os

from flask import Flask

from app.config import Config, LOGGING_CONFIG, LOGS_DIR


def create_app(config_class=Config):
    # Configurar logging unificado usando LOGGING_CONFIG
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        logging.config.dictConfig(LOGGING_CONFIG)
    except Exception as e:
        # Fallback mínimo si falla la configuración
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            f"Fallo al configurar logging con dictConfig: {str(e)}"
        )

    app = Flask(__name__)
    app.config.from_object(config_class)

    # Registrar blueprints
    from app.api.routes import api_bp

    app.register_blueprint(api_bp)

    # Crear directorios si no existen
    for folder in [app.config["UPLOAD_FOLDER"], "models", "results", LOGS_DIR]:
        os.makedirs(folder, exist_ok=True)

    return app
