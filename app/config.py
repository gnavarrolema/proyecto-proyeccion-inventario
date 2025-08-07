import logging
import os
from datetime import datetime
from pathlib import Path

# ======================================================
# CONFIGURACIÓN DE DIRECTORIOS
# ======================================================

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios para datos, modelos y resultados
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Asegurar que los directorios existan
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ======================================================
# CONFIGURACIÓN DE MODO DEPURACIÓN
# ======================================================

# Modo depuración para desarrollo
DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1", "t")

# Nivel de logging basado en modo depuración
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

# Archivo de log con timestamp
current_date = datetime.now().strftime("%Y%m%d")
LOG_FILE = os.path.join(LOGS_DIR, f"app_{current_date}.log")

# ======================================================
# CONFIGURACIÓN DE LA APLICACIÓN FLASK
# ======================================================


class Config:
    """Configuración base para la aplicación Flask."""

    # Configuración de seguridad
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-key-for-development"

    # Configuración de carga de archivos
    UPLOAD_FOLDER = DATA_DIR
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB máximo
    ALLOWED_EXTENSIONS = {"csv"}

    # Configuración de depuración
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
    TESTING = False

    # Configuración de sesión
    SESSION_TYPE = "filesystem"
    PERMANENT_SESSION_LIFETIME = 1800  # 30 minutos

    # Configuración CORS
    CORS_HEADERS = "Content-Type"

    # Timeout para operaciones largas (en segundos)
    OPERATION_TIMEOUT = 300  # 5 minutos


class DevelopmentConfig(Config):
    """Configuración para desarrollo."""

    DEBUG = True
    TESTING = True
    TEMPLATES_AUTO_RELOAD = True


class ProductionConfig(Config):
    """Configuración para producción."""

    DEBUG = False
    TESTING = False

    # En producción, usar una clave secreta fuerte
    SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(24)

    # Configurar SSL en producción
    SSL_REDIRECT = True


# Configuración activa basada en entorno
APP_ENV = os.environ.get("FLASK_ENV", "development")
if APP_ENV == "production":
    app_config = ProductionConfig
else:
    app_config = DevelopmentConfig

# ======================================================
# CONFIGURACIÓN DE MODELOS
# ======================================================

# Parámetros generales para modelos
MODEL_PARAMS = {
    # División entrenamiento/prueba
    "TRAIN_TEST_SPLIT": 0.8,
    # Pasos por defecto a pronosticar
    "DEFAULT_FORECAST_STEPS": 6,
    # Número mínimo de puntos para entrenamiento
    "MIN_TRAIN_POINTS": 12,
    # Columnas objetivo por defecto
    "DEFAULT_TARGET_COLUMNS": ["CANTIDADES", "PRODUCCIÓN"],
    # Validación de pronósticos
    "VALIDATE_FORECASTS": True,
    # Límites para detección de anomalías (multiplicador de desviación estándar)
    "ANOMALY_THRESHOLD": 3.0,
    # Procesamiento paralelo
    "USE_MULTIPROCESSING": True,
    "MAX_WORKERS": os.cpu_count() or 4,
    # Persistencia de modelos
    "SAVE_TRAINED_MODELS": True,
    "REUSE_TRAINED_MODELS": True,
}

# Parámetros para modelo SARIMA
SARIMA_PARAMS = {
    # Orden ARIMA (p,d,q)
    "ORDER": (1, 1, 1),
    # Orden estacional (P,D,Q,s)
    "SEASONAL_ORDER": (1, 1, 1, 12),
    # Enforcing stationarity/invertibility
    "ENFORCE_STATIONARITY": False,
    "ENFORCE_INVERTIBILITY": False,
    # Métodos de estimación
    "METHOD": "css-mle",
    # Información de ajuste
    "DISP": 0 if DEBUG_MODE else -1,
}

# Parámetros para modelo LSTM
LSTM_PARAMS = {
    # Secuencia de entrada
    "LOOKBACK": 12,
    # Arquitectura de red
    "UNITS": [50, 50],  # Unidades en cada capa LSTM
    "DROPOUT": 0.2,
    # Entrenamiento
    "EPOCHS": 100 if not DEBUG_MODE else 10,
    "BATCH_SIZE": 16,
    "VALIDATION_SPLIT": 0.1,
    "EARLY_STOPPING": True,
    "PATIENCE": 10,
    # Optimización
    "OPTIMIZER": "adam",
    "LEARNING_RATE": 0.001,
    # Normalización
    "FEATURE_RANGE": (0, 1),
}

# Parámetros para modelo GRU
GRU_PARAMS = {
    # Secuencia de entrada
    "LOOKBACK": 12,
    # Arquitectura de red
    "UNITS": [50, 50],  # Unidades en cada capa GRU
    "DROPOUT": 0.2,
    # Entrenamiento
    "EPOCHS": 100 if not DEBUG_MODE else 10,
    "BATCH_SIZE": 16,
    "VALIDATION_SPLIT": 0.1,
    "EARLY_STOPPING": True,
    "PATIENCE": 10,
    # Optimización
    "OPTIMIZER": "adam",
    "LEARNING_RATE": 0.001,
    # Normalización
    "FEATURE_RANGE": (0, 1),
}

XGBOOST_PARAMS = {
    # Parámetros del modelo
    "N_ESTIMATORS": 100,
    "LEARNING_RATE": 0.1,
    "MAX_DEPTH": 5,
    # Secuencia de entrada
    "LOOKBACK": 12,
    # Entrenamiento
    "VALIDATION_SPLIT": 0.2,
    # Características adicionales
    "USE_TIME_FEATURES": True,
    # Manejo de estacionalidad
    "SEASONAL_ADJUSTMENT": True,
}

# ======================================================
# CONFIGURACIÓN DE LOGGING
# ======================================================

# Formato de los mensajes de log
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configuración de logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": LOG_FORMAT},
    },
    "handlers": {
        "console": {
            "level": LOG_LEVEL,
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": LOG_LEVEL,
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": True,
        },
        "app": {
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
}

# ======================================================
# CONFIGURACIÓN DE VALIDACIÓN DE DATOS
# ======================================================

# Validación de datos de entrada
DATA_VALIDATION = {
    # Columnas requeridas en CSV
    "REQUIRED_COLUMNS": [
        "Fecha",
        "CENTRO DE COSTO",
        "ARTÍCULO",
        "CANTIDADES",
        "PRODUCCIÓN",
    ],
    # Formato de fecha esperado
    "DATE_FORMAT": "%b-%y",  # ej: 'ene-22'
    # Separador CSV
    "CSV_SEPARATOR": ";",
    # Codificación CSV
    "CSV_ENCODING": "utf-8",
    # Manejo de valores faltantes
    "HANDLE_MISSING": True,
    "MISSING_STRATEGY": "interpolate",  # 'drop', 'interpolate', 'mean', 'median', 'zero'
    # Detección de valores atípicos
    "DETECT_OUTLIERS": True,
    "OUTLIER_THRESHOLD": 3.0,  # IQR multiplier
    "OUTLIER_STRATEGY": "clip",  # 'drop', 'clip', 'winsorize'
}

# ======================================================
# CONFIGURACIÓN DE LA INTERFAZ DE USUARIO
# ======================================================

# Opciones de visualización
UI_CONFIG = {
    # Colores para gráficos
    "CHART_COLORS": {
        "HISTORICAL": "#3498db",
        "SARIMA": "#e74c3c",
        "LSTM": "#2ecc71",
        "GRU": "#9b59b6",
    },
    # Número máximo de puntos a mostrar en gráficos
    "MAX_CHART_POINTS": 50,
    # Formato de exportación
    "EXPORT_FORMATS": ["csv", "json", "excel"],
    # Estilo de tabla
    "TABLE_STYLE": "striped",
    # Paginación
    "PAGINATION_SIZE": 10,
}
