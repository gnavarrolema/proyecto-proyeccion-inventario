import os
import sys

# Añadir el directorio raíz del proyecto al path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, project_root)

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

from app.models.optimization.base_tuner import BaseHyperparameterTuner
from app.models.sarima_model import SarimaModel

# Configurar logger
logger = logging.getLogger(__name__)


class SarimaHyperparameterTuner(BaseHyperparameterTuner):
    # El resto del código...
    """Optimizador de hiperparámetros para modelos SARIMA."""

    def _objective(self, trial, series, cv_folds):
        # Definir espacio de búsqueda
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)
        P = trial.suggest_int("P", 0, 2)
        D = trial.suggest_int("D", 0, 1)
        Q = trial.suggest_int("Q", 0, 2)
        m = trial.suggest_categorical("m", [4, 12])  # Frecuencia estacional

        order = (p, d, q)
        seasonal_order = (P, D, Q, m)

        # Almacenar errores de cada fold
        cv_scores = []

        for train_idx, test_idx in cv_folds:
            # Preparar datos de entrenamiento y prueba
            train_series = pd.Series(
                series.values[train_idx], index=series.index[train_idx]
            )
            test_series = pd.Series(
                series.values[test_idx], index=series.index[test_idx]
            )

            try:
                # Entrenar modelo
                model = SarimaModel(order=order, seasonal_order=seasonal_order)
                model.fit(train_series)

                # Predecir
                forecast = model.predict(steps=len(test_idx))

                # Calcular error
                if len(forecast) == len(test_series):
                    mse = mean_squared_error(
                        test_series.values, forecast.values
                    )
                    cv_scores.append(mse)
                else:
                    cv_scores.append(float("inf"))
            except Exception as e:
                # Añadir valor alto en caso de error
                cv_scores.append(float("inf"))

        # Retornar el MSE promedio (o infinito si todos los folds fallaron)
        if cv_scores:
            mean_mse = np.mean([s for s in cv_scores if s < float("inf")])
            if np.isfinite(mean_mse):
                return mean_mse

        return float("inf")

    def optimize(self, series, article_name=None, progress_callback=None):
        """Optimiza hiperparámetros para SARIMA usando Optuna."""
        logger.info(
            f"Iniciando optimización de SARIMA para serie de {len(series)} puntos"
        )

        if len(series) < 24:
            logger.warning(
                f"Serie temporal insuficiente para optimización: {len(series)} puntos"
            )
            return None

        # Crear folds para validación cruzada
        cv_folds = self._create_cv_folds(series, self.cv_splits)

        # Inicializar estudio de Optuna
        self.study = optuna.create_study(direction="minimize")

        # Contador de trials para seguimiento
        current_trial = 0

        # Función objetivo con manejo de progreso
        def objective_with_progress(trial):
            nonlocal current_trial
            try:
                result = self._objective(trial, series, cv_folds)
                current_trial += 1
                if progress_callback:
                    progress = min(
                        int((current_trial / self.n_trials) * 100), 100
                    )
                    progress_callback(progress, current_trial, self.n_trials)
                return result
            except Exception as e:
                logger.error(f"Error en trial {current_trial}: {str(e)}")
                current_trial += 1
                if progress_callback:
                    progress_callback(
                        int((current_trial / self.n_trials) * 100),
                        current_trial,
                        self.n_trials,
                    )
                return float("inf")

        # Ejecutar optimización
        try:
            self.study.optimize(
                objective_with_progress,
                n_trials=self.n_trials,
                timeout=self.timeout,
            )

            # Guardar resultados
            if hasattr(self.study, "best_params"):
                self.best_params = self.study.best_params
                logger.info(f"Mejores parámetros SARIMA: {self.best_params}")

                # Entrenar con mejores parámetros
                best_model = SarimaModel(
                    order=(
                        self.best_params["p"],
                        self.best_params["d"],
                        self.best_params["q"],
                    ),
                    seasonal_order=(
                        self.best_params["P"],
                        self.best_params["D"],
                        self.best_params["Q"],
                        self.best_params["m"],
                    ),
                )
                best_model.fit(series)

                # Guardar modelo
                if article_name:
                    self._save_model(
                        best_model, self.best_params, article_name, "SARIMA"
                    )

                return self.best_params

            return None
        except Exception as e:
            logger.error(f"Error durante optimización: {str(e)}")
            if progress_callback:
                progress_callback(100, self.n_trials, self.n_trials)
            return None
