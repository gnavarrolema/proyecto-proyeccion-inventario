import logging
import os
import pickle
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class XGBoostHyperparameterTuner:
    """
    Clase para optimizar automáticamente los hiperparámetros de XGBoost para series temporales.
    Utiliza Optuna para realizar optimización bayesiana.
    """

    def __init__(
        self,
        lookback=12,
        n_trials=50,
        cv_splits=3,
        test_size=0.2,
        models_dir="models",
        use_time_features=True,
    ):
        """
        Inicializa el optimizador de hiperparámetros.

        Args:
            lookback: Número de observaciones pasadas a usar como características
            n_trials: Número de combinaciones de hiperparámetros a probar
            cv_splits: Número de divisiones para validación cruzada temporal
            test_size: Fracción de datos a reservar para prueba
            models_dir: Directorio para guardar modelos optimizados
            use_time_features: Si se deben incluir características temporales
        """
        self.lookback = lookback
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.test_size = test_size
        self.models_dir = models_dir
        self.use_time_features = use_time_features
        self.best_params = None
        self.best_model = None
        self.best_score = float("inf")
        self.study = None
        self.feature_importances = None

        # Crear directorio si no existe
        os.makedirs(models_dir, exist_ok=True)

    def _create_features(self, series):
        """
        Crea características para XGBoost a partir de datos históricos.

        Args:
            series: Serie temporal de pandas

        Returns:
            tuple: (X, y) matrices de características y valores objetivo
        """
        # Extraer valores
        values = series.values

        # Crear características básicas (valores anteriores)
        X, y = [], []
        for i in range(len(values) - self.lookback):
            X.append(values[i : i + self.lookback])
            y.append(values[i + self.lookback])

        X = np.array(X)
        y = np.array(y)

        # Añadir características temporales si está habilitado
        if self.use_time_features and hasattr(series.index[0], "month"):
            # Extraer índices de tiempo
            dates = series.index[self.lookback :]

            # Crear características temporales
            month_sin = np.sin(
                2 * np.pi * np.array([d.month for d in dates]) / 12
            )
            month_cos = np.cos(
                2 * np.pi * np.array([d.month for d in dates]) / 12
            )

            # Añadir día del año normalizado
            day_of_year = np.array([d.dayofyear for d in dates]) / 366

            # Quarter del año (trimestre)
            quarter = np.array([(d.month - 1) // 3 + 1 for d in dates]) / 4

            # Tendencia (índice normalizado)
            trend = np.arange(len(dates)) / len(dates)

            # Concatenar características temporales
            time_features = np.column_stack(
                (month_sin, month_cos, day_of_year, quarter, trend)
            )

            # Añadir a la matriz de características
            X_with_time = np.column_stack((X, time_features))
            return X_with_time, y

        return X, y

    def _objective(self, trial, X, y):
        """
        Función objetivo para Optuna que evalúa una combinación de hiperparámetros.

        Args:
            trial: Objeto Trial de Optuna
            X: Matriz de características
            y: Vector de valores objetivo

        Returns:
            float: Error cuadrático medio de validación cruzada
        """
        # Definir espacio de búsqueda de hiperparámetros
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        }

        # TimeSeriesSplit para validación cruzada respetando el orden temporal
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        # Almacenar errores de cada fold
        cv_scores = []

        # Realizar validación cruzada
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Crear y entrenar modelo XGBoost
            model = XGBRegressor(
                objective="reg:squarederror", random_state=42, **params
            )

            # Entrenar modelo
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )

            # Predecir en conjunto de validación
            y_pred = model.predict(X_val)

            # Calcular error
            mse = mean_squared_error(y_val, y_pred)
            cv_scores.append(mse)

        # Retornar el MSE promedio
        mean_mse = np.mean(cv_scores)

        # Guardar el mejor modelo hasta ahora
        if mean_mse < self.best_score:
            self.best_score = mean_mse
            self.best_params = params

            # Entrenar modelo con todos los datos
            self.best_model = XGBRegressor(
                objective="reg:squarederror", random_state=42, **params
            )
            self.best_model.fit(X, y)

            # Guardar importancia de características
            if hasattr(self.best_model, "feature_importances_"):
                self.feature_importances = self.best_model.feature_importances_

        return mean_mse

    def optimize(self, series, article_name=None, progress_callback=None):
        """
        Optimiza los hiperparámetros para una serie temporal dada.

        Args:
            series: Serie temporal de pandas
            article_name: Nombre del artículo (para guardar el modelo)
            progress_callback: Función de callback para reportar progreso

        Returns:
            dict: Mejores hiperparámetros encontrados
        """
        logger.info(
            f"Iniciando optimización de XGBoost para serie de {len(series)} puntos"
        )

        # Preparar datos
        try:
            X, y = self._create_features(series)

            if len(X) < 1 or len(y) < 1:
                logger.error("No hay suficientes datos para la optimización")
                if progress_callback:
                    progress_callback(
                        100, 1, 1
                    )  # Reportar como completado con error
                return None

            # Inicializar estudio de Optuna
            sampler = TPESampler(seed=42)
            self.study = optuna.create_study(
                sampler=sampler, direction="minimize"
            )

            # Variable para rastrear progreso
            self.current_trial = 0

            # Modificar la función objetivo para rastrear progreso con mejor manejo de errores
            original_objective = lambda trial: self._objective(trial, X, y)

            def objective_with_progress(trial):
                try:
                    result = original_objective(trial)
                    self.current_trial += 1
                    if progress_callback:
                        # Calcular progreso real
                        progress = min(
                            int((self.current_trial / self.n_trials) * 100), 100
                        )
                        # Llamada al callback con mejor manejo
                        try:
                            progress_callback(
                                progress, self.current_trial, self.n_trials
                            )
                        except Exception as cb_error:
                            logger.error(
                                f"Error en callback de progreso: {str(cb_error)}"
                            )
                    return result
                except Exception as e:
                    logger.error(
                        f"Error en trial {self.current_trial}: {str(e)}"
                    )
                    self.current_trial += 1
                    if progress_callback:
                        try:
                            progress_callback(
                                int((self.current_trial / self.n_trials) * 100),
                                self.current_trial,
                                self.n_trials,
                            )
                        except:
                            pass
                    # Retornar un valor de error para que Optuna pueda continuar
                    return float("inf")

            # Realizar optimización con un timeout
            try:
                self.study.optimize(
                    objective_with_progress, n_trials=self.n_trials, timeout=600
                )  # 10 min máximo
            except Exception as opt_error:
                logger.error(f"Error en optimización: {str(opt_error)}")
                # Asegurar que reportamos algún progreso incluso si hay error
                if progress_callback and self.current_trial < self.n_trials:
                    try:
                        progress_callback(100, self.n_trials, self.n_trials)
                    except:
                        pass

            # Guardar resultados
            if hasattr(self.study, "best_params"):
                self.best_params = self.study.best_params
                logger.info(
                    f"Mejores hiperparámetros encontrados: {self.best_params}"
                )
                logger.info(
                    f"Mejor puntuación (MSE): {self.study.best_value:.4f}"
                )

                # Guardar modelo si se proporciona nombre de artículo
                if article_name and self.best_model:
                    self._save_model(article_name)

                return self.best_params
            else:
                logger.warning(
                    "No se encontraron mejores parámetros en la optimización"
                )
                return {}

        except Exception as e:
            logger.error(f"Error durante la optimización de XGBoost: {str(e)}")
            # Asegurar que reportamos la finalización incluso si hay error
            if progress_callback:
                try:
                    progress_callback(100, self.n_trials, self.n_trials)
                except:
                    pass
            return None

    def _save_model(self, article_name):
        """
        Guarda el mejor modelo optimizado.

        Args:
            article_name: Nombre del artículo para etiquetar el modelo
        """
        if not self.best_model:
            logger.warning("No hay mejor modelo para guardar")
            return

        # Crear nombre de archivo seguro
        safe_name = (
            article_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xgboost_optimized_{safe_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)

        # Guardar modelo
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.best_model,
                    "params": self.best_params,
                    "score": self.best_score,
                    "feature_importances": self.feature_importances,
                    "lookback": self.lookback,
                    "timestamp": timestamp,
                },
                f,
            )

        logger.info(f"Modelo optimizado guardado en: {filepath}")
        return filepath

    def get_feature_importance(self):
        """
        Obtiene la importancia de las características para el mejor modelo.

        Returns:
            dict: Diccionario con importancia de características
        """
        if not self.feature_importances is not None:
            return {}

        # Crear etiquetas para características
        feature_names = [f"lag_{i+1}" for i in range(self.lookback)]

        # Si usamos características temporales, añadir sus nombres
        if self.use_time_features:
            time_features = [
                "month_sin",
                "month_cos",
                "day_of_year",
                "quarter",
                "trend",
            ]
            feature_names.extend(time_features)

        # Crear diccionario de importancia
        importance_dict = {}
        for i, importance in enumerate(self.feature_importances):
            if i < len(feature_names):
                importance_dict[feature_names[i]] = importance

        # Ordenar por importancia descendente
        return dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

    def plot_optimization_history(self):
        """
        Genera visualizaciones de la optimización (requiere matplotlib).
        """
        if self.study is None:
            logger.warning("No hay estudio de optimización para visualizar")
            return

        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
            )

            # Graficar historia de optimización
            fig1 = plot_optimization_history(self.study)
            fig1.update_layout(title="Historia de optimización")

            # Graficar importancia de parámetros
            fig2 = plot_param_importances(self.study)
            fig2.update_layout(title="Importancia de hiperparámetros")

            return fig1, fig2
        except ImportError:
            logger.warning(
                "Se requiere matplotlib y plotly para visualizaciones"
            )
            return None
