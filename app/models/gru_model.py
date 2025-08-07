# app/models/gru_model.py
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential

logger = logging.getLogger(__name__)


class GRUModel:
    """Implementación del modelo GRU para pronósticos."""

    def __init__(self, lookback=24, units=64, dropout=0.1):
        """
        Inicializa el modelo GRU.

        Args:
            lookback: Número de pasos temporales a considerar
            units: Número de unidades en capas GRU
            dropout: Tasa de dropout para regularización
        """
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _create_sequences(self, data):
        """
        Crea secuencias para entrenamiento GRU.

        Args:
            data: Datos de entrada normalizados

        Returns:
            tuple: (X, y) secuencias de entrada y valores objetivo
        """
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i : i + self.lookback, 0])
            y.append(data[i + self.lookback, 0])

        return np.array(X), np.array(y)

    def _build_model(self, input_shape):
        """
        Construye la arquitectura del modelo GRU.

        Args:
            input_shape: Forma de los datos de entrada

        Returns:
            model: Modelo Keras
        """
        model = Sequential()
        model.add(
            GRU(self.units, return_sequences=True, input_shape=input_shape)
        )
        model.add(Dropout(self.dropout))
        model.add(GRU(self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")
        return model

    def _normalize_data_robust(self, values):
        """
        Normaliza datos de manera robusta, detectando y manejando valores atípicos.

        Args:
            values: Valores a normalizar

        Returns:
            array: Valores normalizados
        """
        # Calcular estadísticas para detección de valores atípicos
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        # Definir límites para valores atípicos (usando regla 1.5*IQR)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Log para diagnóstico
        logger.info(
            f"Estadísticas para normalización GRU: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}"
        )
        logger.info(
            f"Límites para valores atípicos: [{lower_bound:.2f}, {upper_bound:.2f}]"
        )
        logger.info(
            f"Rango original: [{np.min(values):.2f}, {np.max(values):.2f}]"
        )

        # Recortar valores extremos para el cálculo de escala
        values_for_scaling = np.clip(values, lower_bound, upper_bound)

        # Ajustar scaler con valores recortados (para que la escala no se distorsione por extremos)
        self.scaler.fit(values_for_scaling)

        # Log para diagnóstico
        logger.info(
            f"Rango de scaler GRU: [{self.scaler.data_min_[0]:.2f}, {self.scaler.data_max_[0]:.2f}]"
        )

        # Transformar los valores originales
        scaled = self.scaler.transform(values)

        return scaled

    def _get_monthly_seasonality(self, series):
        """
        Calcula factores de estacionalidad mensuales aproximados.

        Args:
            series: Serie temporal con más de 12 observaciones

        Returns:
            dict: Factores de estacionalidad por mes
        """
        seasonal_factors = {}

        try:
            # Solo intentar si hay al menos 12 observaciones
            if len(series) >= 12:
                # Obtener los índices de la serie y extraer meses
                dates = series.index
                months = [d.month if hasattr(d, "month") else 1 for d in dates]

                # Calcular promedio por mes
                month_data = {}
                for i, val in enumerate(series):
                    month = months[i]
                    if month not in month_data:
                        month_data[month] = []
                    month_data[month].append(val)

                # Calcular promedio de toda la serie
                overall_mean = series.mean()

                # Calcular factores estacionales
                for month, values in month_data.items():
                    month_mean = np.mean(values)
                    seasonal_factors[month] = (
                        month_mean / overall_mean if overall_mean > 0 else 1.0
                    )

                # Estabilizar factores extremos pero permitir un poco más de variación
                for month in seasonal_factors:
                    if (
                        seasonal_factors[month] > 1.08
                    ):  # Permitir hasta 8% por encima
                        seasonal_factors[month] = 1.08
                    elif (
                        seasonal_factors[month] < 0.92
                    ):  # Permitir hasta 8% por debajo
                        seasonal_factors[month] = 0.92

                logger.info(
                    f"Factores estacionales por mes (GRU): {seasonal_factors}"
                )
            else:
                # Si no hay suficientes datos, usar factores neutrales
                for month in range(1, 13):
                    seasonal_factors[month] = 1.0
        except Exception as e:
            logger.warning(f"Error al calcular estacionalidad: {str(e)}")
            # Usar factores neutrales en caso de error
            for month in range(1, 13):
                seasonal_factors[month] = 1.0

        return seasonal_factors

    def _analyze_patterns(self, series):
        """
        Analiza patrones cíclicos en la serie temporal.

        Args:
            series: Serie temporal

        Returns:
            dict: Información sobre patrones detectados
        """
        pattern_info = {"detected": False}

        try:
            # Calcular autocorrelaciones para detectar patrones
            if len(series) >= 24:
                # Extraer valores
                values = series.values

                # Calcular diferencias de primer orden
                diffs = np.diff(values)

                # Contar cambios de dirección (cruces por cero)
                direction_changes = np.sum(np.diff(np.signbit(diffs)))

                # Normalizar por longitud
                cycle_frequency = direction_changes / len(diffs)

                # Estimar periodicidad aproximada
                if cycle_frequency > 0:
                    estimated_period = int(1 / cycle_frequency)
                    if 2 <= estimated_period <= 12:  # Períodos razonables
                        pattern_info["detected"] = True
                        pattern_info["period"] = estimated_period
                        pattern_info["intensity"] = cycle_frequency
                        logger.info(
                            f"Patrón cíclico detectado: período ~{estimated_period}, intensidad: {cycle_frequency:.3f}"
                        )
        except Exception as e:
            logger.warning(f"Error al analizar patrones: {str(e)}")

        return pattern_info

    def fit(self, series, epochs=100, batch_size=16, validation_split=0.2):
        """
        Entrena el modelo con los datos.

        Args:
            series: Serie temporal para entrenamiento
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del lote para entrenamiento
            validation_split: Fracción de datos para validación

        Returns:
            history: Historial de entrenamiento
        """
        logger.info(f"Entrenando modelo GRU con {len(series)} puntos")
        try:
            # --- INICIO DE LA CORRECCIÓN ROBUSTA ---

            # 1. Dividir la serie para encontrar los datos de entrenamiento y ajustar el scaler
            # Esto evita el data leakage, el scaler solo "aprende" de los datos de entrenamiento.
            split_index = int(len(series) * (1 - validation_split))
            train_values = series.values[:split_index].reshape(-1, 1)
            
            if len(train_values) == 0:
                raise ValueError("No hay suficientes datos para el conjunto de entrenamiento después de la división.")

            # 2. Ajustar el scaler SOLO con los datos de entrenamiento
            self.scaler.fit(train_values)

            # 3. Transformar la serie COMPLETA con el scaler ya ajustado
            scaled_data = self.scaler.transform(series.values.reshape(-1, 1))

            # 4. Crear secuencias a partir de TODOS los datos escalados
            X, y = self._create_sequences(scaled_data)

            if len(X) == 0:
                raise ValueError(f"No se pudieron crear secuencias. La longitud de la serie ({len(series)}) podría ser menor que el lookback ({self.lookback}).")

            # Reshape para GRU
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # --- FIN DE LA CORRECCIÓN ROBUSTA ---

            # Calcular factores estacionales aproximados (para uso en predicción)
            self.seasonal_factors = self._get_monthly_seasonality(series)

            # Analizar patrones cíclicos
            self.pattern_info = self._analyze_patterns(series)

            # Construir modelo
            self.model = self._build_model((X.shape[1], 1))

            # Configurar early stopping para evitar sobreajuste
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            )

            # Entrenar modelo
            history = self.model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split, # Keras se encarga de la división en los datos ya procesados
                callbacks=[early_stopping],
                verbose=1,
            )

            logger.info("Modelo GRU entrenado correctamente")
            return history
        except Exception as e:
            logger.error(f"Error en entrenamiento de GRU: {str(e)}")
            raise

    def predict(self, series, steps=6):
        """
        Realiza pronósticos con el modelo GRU.

        Args:
            series: Serie temporal (últimos valores)
            steps: Número de pasos a pronosticar

        Returns:
            np.array: Pronósticos
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir")

        try:
            # Calcular estadísticas de la serie para validación posterior
            recent_data = series[-24:] if len(series) > 24 else series
            recent_mean = recent_data.mean()
            recent_std = recent_data.std()
            recent_max = recent_data.max()
            recent_min = recent_data.min()
            global_max = series.max()
            global_min = series.min()

            # Obtener el último valor conocido
            last_known_value = (
                series.iloc[-1] if len(series) > 0 else recent_mean
            )

            # Log para diagnóstico
            logger.info(
                f"Estadísticas recientes para validación GRU - Media: {recent_mean:.2f}, Std: {recent_std:.2f}, Min: {recent_min:.2f}, Max: {recent_max:.2f}"
            )
            logger.info(
                f"Estadísticas globales - Min: {global_min:.2f}, Max: {global_max:.2f}"
            )
            logger.info(f"Último valor conocido: {last_known_value:.2f}")

            # Obtener los últimos valores para la predicción
            last_sequence = series[-self.lookback :].values.reshape(-1, 1)

            # Normalizar secuencia
            last_sequence = self.scaler.transform(last_sequence)

            # Preparar entrada para el modelo
            current_sequence = last_sequence.reshape(1, self.lookback, 1)

            # Obtener el último mes conocido
            last_date = series.index[-1]

            # Calcular tendencia reciente (últimos 6 valores)
            trend_factor = 0
            if len(series) >= 6:
                recent_window = series[-6:]
                if len(recent_window) > 1:
                    # Calcular pendiente simple
                    x = np.arange(len(recent_window))
                    y = recent_window.values

                    if len(x) == len(y) and len(x) > 1:
                        # Ajuste lineal simple
                        slope = np.polyfit(x, y, 1)[0]
                        # Normalizar la pendiente como porcentaje del último valor
                        if last_known_value != 0:
                            trend_factor = slope / last_known_value
                            # Limitar el factor de tendencia a máximo 3% por mes
                            trend_factor = max(min(trend_factor, 0.03), -0.03)

                        logger.info(
                            f"Pendiente calculada GRU: {slope:.2f}, Factor de tendencia: {trend_factor:.4f}"
                        )

            # Obtener patrón cíclico si existe
            has_pattern = getattr(self, "pattern_info", {}).get(
                "detected", False
            )
            pattern_period = getattr(self, "pattern_info", {}).get("period", 6)

            # Crear suavizado mediante promedio móvil para las predicciones
            window_size = 3
            smoothed_values = []

            # Predicciones iterativas
            raw_predictions = []

            # Precalcular factores cíclicos si hay patrón
            cyclic_factors = {}
            if has_pattern:
                base_amplitude = (
                    recent_std * 0.15
                )  # 15% de la desviación estándar
                # Crear factores de ciclo como una onda sinusoidal suave
                for i in range(steps):
                    phase = 2 * np.pi * (i % pattern_period) / pattern_period
                    cyclic_factors[i] = (
                        1 + base_amplitude * np.sin(phase) / recent_mean
                    )

            for i in range(steps):
                # Determinar el mes de la predicción
                if hasattr(last_date, "month"):
                    # Calcular el mes de predicción (añadir i+1 meses)
                    if hasattr(last_date, "replace"):  # pandas Timestamp
                        try:
                            pred_date = last_date + pd.DateOffset(months=i + 1)
                            pred_month = pred_date.month
                        except:
                            pred_month = ((last_date.month + i) % 12) + 1
                    else:  # datetime
                        pred_month = ((last_date.month + i) % 12) + 1
                else:
                    # Si no hay fecha, usar valor neutro
                    pred_month = 1

                # Predecir el siguiente valor
                pred = self.model.predict(current_sequence, verbose=0)
                pred_value = pred[0, 0]

                # Aplicar un factor estacional
                seasonal_factor = getattr(self, "seasonal_factors", {}).get(
                    pred_month, 1.0
                )

                # Aplicar el efecto estacional (20% del efecto calculado)
                seasonal_effect = 1.0 + (seasonal_factor - 1.0) * 0.20

                # Aplicar factor cíclico si existe
                cyclic_effect = 1.0
                if has_pattern:
                    cyclic_effect = cyclic_factors.get(i, 1.0)

                # Aplicar el efecto de tendencia
                # El efecto aumenta gradualmente y se detiene después de 3 meses
                if i < 3:
                    trend_effect = 1.0 + trend_factor * (i + 1)
                else:
                    trend_effect = 1.0 + trend_factor * 3

                # Combinar efectos y aplicar al valor predicho
                adjusted_pred = (
                    pred_value * seasonal_effect * trend_effect * cyclic_effect
                )

                # Añadir a las predicciones crudas
                raw_predictions.append(adjusted_pred)

                # Actualizar la secuencia para la siguiente iteración
                new_point = np.array([adjusted_pred]).reshape(1, 1, 1)
                current_sequence = np.append(
                    current_sequence[:, 1:, :], new_point, axis=1
                )

            # Aplicar suavizado a las predicciones
            for i in range(len(raw_predictions)):
                if i < window_size - 1:
                    # Para los primeros puntos, usar promedio con los valores disponibles
                    window = raw_predictions[: i + 1]
                    smoothed_value = np.mean(window)
                else:
                    # Para el resto, usar ventana completa
                    window = raw_predictions[i - (window_size - 1) : i + 1]
                    smoothed_value = np.mean(window)

                smoothed_values.append(smoothed_value)

            # Convertir predicciones a array y desnormalizar
            smoothed_values = np.array(smoothed_values).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(smoothed_values)

            # Log del primer valor pronosticado
            logger.info(
                f"Primer valor pronosticado GRU: {predictions[0][0]:.2f}"
            )

            # Validar pronósticos (límites razonables)
            # Ajustar los límites para que sean más estrictos
            upper_limit = min(
                global_max * 1.15,
                recent_max * 1.15,
                recent_mean + 1.5 * recent_std,
            )
            lower_limit = max(
                global_min * 0.85,
                recent_min * 0.85,
                recent_mean - 1.5 * recent_std,
                0,
            )

            logger.info(
                f"Límites de validación GRU - Inferior: {lower_limit:.2f}, Superior: {upper_limit:.2f}"
            )

            # Ajustar valores fuera de rango
            for i in range(len(predictions)):
                if predictions[i, 0] > upper_limit:
                    logger.warning(
                        f"Ajustando pronóstico GRU alto: {predictions[i, 0]:.2f} → {upper_limit:.2f}"
                    )
                    predictions[i, 0] = upper_limit
                elif predictions[i, 0] < lower_limit:
                    logger.warning(
                        f"Ajustando pronóstico GRU bajo: {predictions[i, 0]:.2f} → {lower_limit:.2f}"
                    )
                    predictions[i, 0] = lower_limit

            # Asegurar que haya variabilidad entre puntos consecutivos
            # Esto evita líneas perfectamente planas
            for i in range(1, len(predictions)):
                # Si dos puntos consecutivos son demasiado similares, añadir variación
                if (
                    abs(predictions[i, 0] - predictions[i - 1, 0])
                    < recent_std * 0.05
                ):
                    # Añadir variación basada en la desviación estándar
                    variation = np.random.normal(0, recent_std * 0.03)
                    old_value = predictions[i, 0]
                    predictions[i, 0] = predictions[i, 0] + variation

                    # Asegurar que sigue dentro de los límites
                    if predictions[i, 0] > upper_limit:
                        predictions[i, 0] = upper_limit
                    elif predictions[i, 0] < lower_limit:
                        predictions[i, 0] = lower_limit

                    logger.info(
                        f"Añadida variación al punto {i}: {old_value:.2f} -> {predictions[i, 0]:.2f}"
                    )

            # Suavizar transición entre último valor histórico y primero pronosticado
            if len(predictions) > 0:
                # Ajustar el primer valor pronosticado para una transición más suave
                first_pred = predictions[0, 0]
                # Mover el primer valor 70% hacia el último conocido
                blend_factor = 0.3  # Conservar solo 30% del valor predicho
                blended_first = (
                    last_known_value * (1 - blend_factor)
                    + first_pred * blend_factor
                )

                # Log del ajuste
                logger.info(
                    f"Ajustando primera predicción GRU de {first_pred:.2f} a {blended_first:.2f} para suavizar transición"
                )

                # Aplicar el ajuste
                predictions[0, 0] = blended_first

            return predictions.flatten()
        except Exception as e:
            logger.error(f"Error al generar pronóstico GRU: {str(e)}")
            raise
