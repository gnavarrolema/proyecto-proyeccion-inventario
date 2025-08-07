# app/models/xgboost_model.py
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class XGBoostModel:
    """Implementación simplificada del modelo XGBoost para pronósticos."""

    def __init__(
        self, lookback=12, n_estimators=100, learning_rate=0.1, max_depth=5
    ):
        """
        Inicializa el modelo XGBoost.

        Args:
            lookback: Número de pasos temporales a considerar
            n_estimators: Número de árboles en el modelo
            learning_rate: Tasa de aprendizaje
            max_depth: Profundidad máxima de cada árbol
        """
        self.lookback = lookback
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _prepare_features(self, series):
        """
        Prepara las características para el modelo.

        Args:
            series: Serie temporal pandas

        Returns:
            tuple: X (características), y (target) y series escalada
        """
        # Escalar los datos
        values = series.values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values).flatten()

        # Crear características (usamos solo valores pasados como características)
        X, y = [], []
        for i in range(len(scaled_values) - self.lookback):
            X.append(scaled_values[i : i + self.lookback])
            y.append(scaled_values[i + self.lookback])

        if not X:  # Si no hay suficientes datos
            return np.array([]), np.array([]), scaled_values

        return np.array(X), np.array(y), scaled_values

    def fit(self, series):
        """
        Entrena el modelo con la serie temporal.

        Args:
            series: Serie temporal para entrenamiento

        Returns:
            self: Modelo entrenado
        """
        logger.info(f"Entrenando modelo XGBoost con {len(series)} puntos")

        try:
            # Preparar características
            X, y, _ = self._prepare_features(series)

            if len(X) == 0:
                logger.warning(
                    "No hay suficientes datos para entrenar el modelo XGBoost"
                )
                return self

            # Crear y entrenar modelo
            self.model = XGBRegressor(**self.params)
            self.model.fit(X, y)

            logger.info(f"Modelo XGBoost entrenado con {len(X)} muestras")
            return self

        except Exception as e:
            logger.error(f"Error al entrenar modelo XGBoost: {str(e)}")
            raise

    def predict(self, series, steps=6):
        """
        Genera pronósticos.

        Args:
            series: Serie temporal para la predicción
            steps: Número de pasos a pronosticar

        Returns:
            array: Pronósticos para los próximos 'steps' períodos
        """
        if self.model is None:
            logger.error("El modelo XGBoost no ha sido entrenado")
            return np.zeros(steps)

        try:
            # Obtener estadísticas para validación posterior
            recent_data = series[-24:] if len(series) > 24 else series
            stats = {
                "mean": recent_data.mean(),
                "std": recent_data.std(),
                "min": recent_data.min(),
                "max": recent_data.max(),
                "last": series.iloc[-1]
                if len(series) > 0
                else recent_data.mean(),
            }

            # Escalar datos
            values = series.values.reshape(-1, 1)
            scaled_values = self.scaler.transform(values).flatten()

            # Preparar datos para la predicción
            input_sequence = scaled_values[-self.lookback :].copy()

            # Array para almacenar pronósticos
            forecasts = []

            # Generar pronósticos paso a paso
            for i in range(steps):
                # Hacer la predicción
                try:
                    pred = self.model.predict(input_sequence.reshape(1, -1))[0]
                except Exception as e:
                    logger.warning(f"Error en predicción individual: {str(e)}")
                    # Si hay error, usar valor anterior o media
                    pred = input_sequence[-1]

                # Guardar predicción
                forecasts.append(pred)

                # Actualizar secuencia para siguiente paso
                input_sequence = np.append(input_sequence[1:], pred)

            # Convertir a array y desnormalizar
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts = self.scaler.inverse_transform(forecasts).flatten()

            # Validar pronósticos (límites razonables)
            upper_limit = stats["max"] * 1.2
            lower_limit = max(0, stats["min"] * 0.8)

            # Ajustar valores fuera de rango
            for i in range(len(forecasts)):
                if forecasts[i] > upper_limit:
                    forecasts[i] = upper_limit
                elif forecasts[i] < lower_limit:
                    forecasts[i] = lower_limit

            # Suavizar primer valor para transición más gradual
            if len(forecasts) > 0:
                forecasts[0] = 0.7 * stats["last"] + 0.3 * forecasts[0]

            return forecasts

        except Exception as e:
            logger.error(f"Error en predicción XGBoost: {str(e)}")
            # Devolver array de ceros como fallback
            return np.zeros(steps)
