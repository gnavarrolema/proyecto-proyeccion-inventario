# app/models/sarima_model.py
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


class SarimaModel:
    """Implementación del modelo SARIMA para pronósticos."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Inicializa el modelo SARIMA.

        Args:
            order: Orden del modelo (p,d,q)
            seasonal_order: Orden estacional (P,D,Q,s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None

    def fit(self, series):
        """
        Entrena el modelo con los datos.

        Args:
            series: Serie temporal para entrenamiento

        Returns:
            self: Modelo entrenado
        """
        logger.info(f"Entrenando modelo SARIMA con {len(series)} puntos")
        try:
            self.model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            self.fitted_model = self.model.fit(disp=False)
            logger.info("Modelo SARIMA entrenado correctamente")
            return self
        except Exception as e:
            logger.error(f"Error en entrenamiento de SARIMA: {str(e)}")
            raise

    def predict(self, steps=6):
        """
        Realiza pronósticos.

        Args:
            steps: Número de pasos a pronosticar

        Returns:
            pd.Series: Pronósticos
        """
        if self.fitted_model is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir")

        try:
            forecast = self.fitted_model.get_forecast(steps=steps)
            pred_mean = forecast.predicted_mean

            logger.info(f"Pronóstico SARIMA generado para {steps} pasos")
            return pred_mean
        except Exception as e:
            logger.error(f"Error al generar pronóstico SARIMA: {str(e)}")
            raise

    def get_summary(self):
        """Obtiene resumen del modelo entrenado."""
        if self.fitted_model is None:
            return "Modelo no entrenado"

        return self.fitted_model.summary()