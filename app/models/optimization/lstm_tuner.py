# app/models/optimization/lstm_tuner.py
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

from app.models.lstm_model import LSTMModel
from app.models.optimization.base_tuner import BaseHyperparameterTuner

# Configurar logger
logger = logging.getLogger(__name__)


class LSTMHyperparameterTuner(BaseHyperparameterTuner):
    # Resto del código...
    """Optimizador de hiperparámetros para modelos LSTM."""

    def _objective(self, trial, series, cv_folds):
        # Definir espacio de búsqueda
        lookback = trial.suggest_int("lookback", 6, 24)
        units = trial.suggest_int("units", 16, 128)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

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

            if len(train_series) <= lookback:
                # Skip si no hay suficientes datos
                continue

            try:
                # Entrenar modelo
                model = LSTMModel(
                    lookback=lookback, units=units, dropout=dropout
                )
                model.fit(
                    train_series,
                    epochs=30,  # Reducido para optimización
                    batch_size=16,
                    validation_split=0.1,
                )

                # Predecir
                forecast = model.predict(train_series, steps=len(test_idx))

                # Calcular error
                if len(forecast) == len(test_series):
                    mse = mean_squared_error(test_series.values, forecast)
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
        """Optimiza hiperparámetros para LSTM usando Optuna."""
        logger.info(
            f"Iniciando optimización de LSTM para serie de {len(series)} puntos"
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
                logger.info(f"Mejores parámetros LSTM: {self.best_params}")

                # Entrenar con mejores parámetros
                best_model = LSTMModel(
                    lookback=self.best_params["lookback"],
                    units=self.best_params["units"],
                    dropout=self.best_params["dropout"],
                )
                best_model.fit(series)

                # Guardar modelo
                if article_name:
                    self._save_model(
                        best_model, self.best_params, article_name, "LSTM"
                    )

                return self.best_params

            return None
        except Exception as e:
            logger.error(f"Error durante optimización: {str(e)}")
            if progress_callback:
                progress_callback(100, self.n_trials, self.n_trials)
            return None