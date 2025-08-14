import logging

import numpy as np
import optuna

logger = logging.getLogger(__name__)


class BaseHyperparameterTuner:
    """Clase base para optimización de hiperparámetros."""

    def __init__(
        self, n_trials=50, timeout=600, cv_splits=3, models_dir="models"
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_splits = cv_splits
        self.models_dir = models_dir
        self.best_params = None
        self.best_score = float("inf")
        self.study = None

    def optimize(self, series, article_name=None, progress_callback=None):
        """
        Método abstracto para optimizar hiperparámetros.
        Debe ser implementado por subclases.
        """
        raise NotImplementedError("Las subclases deben implementar este método")

    def _create_cv_folds(self, series, n_splits=3):
        """Crea folds para validación cruzada respetando el orden temporal."""
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        data = series.values
        return [
            (train_idx, test_idx) for train_idx, test_idx in tscv.split(data)
        ]

    def _save_model(self, model, params, article_name, model_type):
        """Guarda el modelo optimizado."""
        import os
        import pickle
        from datetime import datetime

        if model is None:
            return None

        # Crear nombre de archivo seguro
        safe_name = (
            article_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_optimized_{safe_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)

        # Guardar modelo
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "params": params,
                    "score": self.best_score,
                    "timestamp": timestamp,
                },
                f,
            )

        logger.info(f"Modelo optimizado guardado en: {filepath}")
        return filepath