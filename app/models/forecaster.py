import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unicodedata
from sklearn.model_selection import TimeSeriesSplit


def create_temporal_split(series, validation_split=0.2, min_train_samples=10):
    """
    Crea una división temporal robusta para series temporales evitando data leakage.
    
    Args:
        series: Serie temporal pandas
        validation_split: Fracción para validación (0.0-1.0)
        min_train_samples: Mínimo número de muestras para entrenamiento
    
    Returns:
        tuple: (train_series, val_series, split_info)
    """
    total_length = len(series)
    
    # Para series muy pequeñas, usar división mínima
    if total_length < 20:
        train_size = max(min_train_samples, int(total_length * (1 - validation_split)))
        if train_size >= total_length - 1:
            train_size = total_length - 2
        
        train_series = series[:train_size]
        val_series = series[train_size:]
        
        split_info = {
            'method': 'simple',
            'total_samples': total_length,
            'train_samples': len(train_series),
            'val_samples': len(val_series)
        }
        
    else:
        # Para series más largas, usar TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=min(3, total_length // 10))  # Máximo 3 splits
        splits = list(tscv.split(series))
        
        if len(splits) > 0:
            # Usar la última división (más datos para entrenamiento)
            train_idx, val_idx = splits[-1]
            train_series = series.iloc[train_idx]
            val_series = series.iloc[val_idx]
        else:
            # Fallback a división simple
            train_size = int(total_length * (1 - validation_split))
            train_series = series[:train_size]
            val_series = series[train_size:]
        
        split_info = {
            'method': 'time_series_split',
            'total_samples': total_length,
            'train_samples': len(train_series),
            'val_samples': len(val_series),
            'n_splits': len(splits)
        }
    
    return train_series, val_series, split_info


def validate_temporal_integrity(train_series, val_series, forecast_dates=None):
    """
    Valida que no haya data leakage temporal en las divisiones de datos.
    
    Args:
        train_series: Serie de entrenamiento
        val_series: Serie de validación  
        forecast_dates: Fechas de pronóstico (opcional)
    
    Returns:
        dict: Resultado de la validación
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Verificar que las fechas de entrenamiento sean anteriores a las de validación
    if len(train_series) > 0 and len(val_series) > 0:
        last_train_date = train_series.index[-1]
        first_val_date = val_series.index[0]
        
        if last_train_date >= first_val_date:
            validation_result['errors'].append(
                f"Data leakage detectado: última fecha entrenamiento ({last_train_date}) >= primera fecha validación ({first_val_date})"
            )
            validation_result['is_valid'] = False
    
    # Verificar que las fechas de pronóstico sean posteriores a los datos
    if forecast_dates is not None and len(val_series) > 0:
        last_data_date = val_series.index[-1]
        first_forecast_date = forecast_dates[0] if len(forecast_dates) > 0 else None
        
        if first_forecast_date and first_forecast_date <= last_data_date:
            validation_result['warnings'].append(
                f"Pronóstico para fecha pasada: primera fecha pronóstico ({first_forecast_date}) <= última fecha datos ({last_data_date})"
            )
    
    # Verificar continuidad temporal
    for name, series in [('entrenamiento', train_series), ('validación', val_series)]:
        if len(series) > 1:
            # Verificar que las fechas estén ordenadas
            if not series.index.is_monotonic_increasing:
                validation_result['warnings'].append(
                    f"Fechas no ordenadas en conjunto de {name}"
                )
    
    return validation_result

from app.config import MODELS_DIR, RESULTS_DIR
from app.models.gru_model import GRUModel
from app.models.lstm_model import LSTMModel
from app.models.safety_stock import SafetyStockCalculator
from app.models.sarima_model import SarimaModel
from app.models.xgboost_model import XGBoostModel
from app.utils.data_loader import DataLoader
from app.utils.leadtime_handler import LeadTimeHandler
from app.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class Forecaster:
    """Clase principal para pronósticos de inventario."""

    def __init__(self):
        """Inicializa el forecaster."""
        self.data = None
        self.articles = None
        self.results = {}
        self.models = {}
        self.leadtime_handler = LeadTimeHandler()
        self.safety_stock_calculator = SafetyStockCalculator()

    def _convert_date_format(self, date_str):
        """
        Convierte formato de fecha de 'mes-año' a 'año-mes-01'.

        Args:
            date_str (str): Fecha en formato 'mes-año'

        Returns:
            str: Fecha en formato 'año-mes-01'
        """
        if not isinstance(date_str, str):
            return None

        parts = date_str.split("-")
        if len(parts) != 2:
            return None

        month_map = {
            "ene": "01",
            "feb": "02",
            "mar": "03",
            "abr": "04",
            "may": "05",
            "jun": "06",
            "jul": "07",
            "ago": "08",
            "sept": "09",
            "sep": "09",
            "oct": "10",
            "nov": "11",
            "dic": "12",
        }

        month = month_map.get(parts[0].lower())
        if not month:
            return None

        year = parts[1] if len(parts[1]) == 4 else "20" + parts[1]

        return f"{year}-{month}-01"

    def load_data(self, file_path):
        """
        Carga y preprocesa datos desde un archivo CSV o Excel.

        Args:
            file_path: Ruta al archivo CSV/XLSX/XLS

        Returns:
            pd.DataFrame: Datos procesados
        """
        logger.info(f"Cargando datos desde {file_path}")

        # Verificar si el archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encuentra el archivo: {file_path}")

        # Detectar extensión y leer en consecuencia
        from pathlib import Path as _Path

        ext = _Path(file_path).suffix.lower()
        if ext in {".xlsx", ".xls"}:
            raw_data = pd.read_excel(file_path)
        else:
            # CSV con formato español/latino: decimal=',' y thousands='.'
            raw_data = pd.read_csv(file_path, sep=";", decimal=",", thousands=".")

        # Normalizar nombres de columnas para manejar variantes comunes
        def _strip_accents(s):
            if not isinstance(s, str):
                return s
            return "".join(
                c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
            )

        def _normalize_colname(s):
            s0 = _strip_accents(str(s)).strip().lower()
            s0 = s0.replace(" ", "")
            s0 = s0.replace("_", "")
            s0 = s0.replace("&", "y")
            return s0

        canonical_map = {
            # Fecha / Mes&Año
            "mesyano": "Mes&Año",
            "mesano": "Mes&Año",
            "mesyaño": "Mes&Año",
            "fecha": "Fecha",
            # Artículo
            "articulo": "ARTÍCULO",
            "articulo": "ARTÍCULO",
            "articulos": "ARTÍCULO",
            "producto": "ARTÍCULO",
            # Centro de costo
            "centrodecosto": "CENTRO DE COSTO",
            # Cantidades
            "cantidad": "CANTIDADES",
            "cantidades": "CANTIDADES",
            "unidades": "CANTIDADES",
            # Producción
            "produccion": "PRODUCCIÓN",
            "producciones": "PRODUCCIÓN",
        }

        rename_dict = {}
        for col in list(raw_data.columns):
            key = _normalize_colname(col)
            if key in canonical_map:
                rename_dict[col] = canonical_map[key]
        if rename_dict:
            raw_data = raw_data.rename(columns=rename_dict)

        # Verificar que cargamos correctamente los datos
        try:
            logger.info(f"Muestra de datos cargados para verificación:")
            sample_article = raw_data["ARTÍCULO"].iloc[0]
            sample_data = raw_data[raw_data["ARTÍCULO"] == sample_article].head(
                3
            )
            for _, row in sample_data.iterrows():
                logger.info(
                    f"  {row['Mes&Año']} - {sample_article}: CANTIDADES={row['CANTIDADES']}, PRODUCCIÓN={row['PRODUCCIÓN']}"
                )
        except Exception as e:
            logger.warning(f"No se pudo mostrar muestra de datos: {str(e)}")

        # Asignar los datos cargados directamente
        # NOTA: Si tu clase Forecaster tiene algún proceso adicional después de cargar,
        # necesitarás adaptarlo aquí según tu implementación actual
        self.data = raw_data

        # Procesar fechas si es necesario
        if "Mes&Año" in self.data.columns:
            col = self.data["Mes&Año"]
            try:
                import pandas as _pd
                if _pd.api.types.is_datetime64_any_dtype(col):
                    parsed = _pd.to_datetime(col, errors="coerce")
                    self.data["fecha_std"] = parsed.dt.to_period("M").dt.to_timestamp()
                else:
                    # Si viene como string tipo 'ene-22'
                    self.data["fecha_std"] = col.apply(self._convert_date_format)
            except Exception:
                # Fallback robusto
                self.data["fecha_std"] = _pd.to_datetime(col, errors="coerce").dt.to_period("M").dt.to_timestamp()
        elif "Fecha" in self.data.columns:
            # Intentar parsear directamente y ajustar al inicio de mes
            parsed = pd.to_datetime(self.data["Fecha"], errors="coerce", dayfirst=True)
            # Normalizar al primer día del mes
            self.data["fecha_std"] = parsed.dt.to_period("M").dt.to_timestamp()

        # Asegurar columnas numéricas limpias si existen
        for num_col in ["CANTIDADES", "PRODUCCIÓN"]:
            if num_col in self.data.columns and self.data[num_col].dtype == object:
                cleaned = (
                    self.data[num_col]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                self.data[num_col] = pd.to_numeric(cleaned, errors="coerce")

        # Extraer lista de artículos únicos si la columna existe
        if "ARTÍCULO" in self.data.columns:
            self.articles = self.data["ARTÍCULO"].dropna().astype(str).unique()
        else:
            # Fallback: intentar columnas alternativas conocidas
            alt_cols = [c for c in self.data.columns if "ART" in _strip_accents(c).upper()]
            if alt_cols:
                self.articles = self.data[alt_cols[0]].dropna().astype(str).unique()
            else:
                self.articles = []

        logger.info(
            f"Datos cargados: {len(self.data)} registros, {len(self.articles)} artículos"
        )

        return self.data

    def prepare_time_series(self, article, target="CANTIDADES"):
        """
        Prepara serie temporal para un artículo específico.

        Args:
            article: Nombre del artículo
            target: Columna objetivo ('CANTIDADES' o 'PRODUCCIÓN')

        Returns:
            pd.Series: Serie temporal del artículo
        """
        if self.data is None:
            raise ValueError("Primero debe cargar los datos con load_data()")

        # Filtrar por artículo
        article_data = self.data[self.data["ARTÍCULO"] == article].copy()

        # Verificar si hay datos
        if len(article_data) == 0:
            logger.warning(
                f"No se encontraron datos para el artículo: {article}"
            )
            return pd.Series()

        # Verificar si la columna objetivo existe
        if target not in article_data.columns:
            raise ValueError(f"La columna {target} no existe en los datos")

        # Verificar si tenemos columna fecha_std
        if "fecha_std" in article_data.columns:
            # Asegurarnos que fecha_std sea datetime
            article_data["fecha_std"] = pd.to_datetime(
                article_data["fecha_std"]
            )
        else:
            # Si no existe fecha_std, intentar crearla desde Mes&Año
            if "Mes&Año" in article_data.columns:
                article_data["fecha_std"] = article_data["Mes&Año"].apply(
                    lambda x: pd.to_datetime(self._convert_date_format(x))
                )

        # Verificar si tenemos fecha_std
        if (
            "fecha_std" not in article_data.columns
            or article_data["fecha_std"].isna().all()
        ):
            logger.error(
                f"No se pudo crear la columna fecha_std para {article}"
            )
            return pd.Series()

        # Eliminar duplicados y ordenar por fecha
        article_data = article_data.drop_duplicates(
            subset=["fecha_std"]
        ).sort_values("fecha_std")

        # Crear serie temporal indexada por fecha
        series = pd.Series(
            article_data[target].values, index=article_data["fecha_std"]
        )

        logger.info(
            f"Serie temporal preparada para {article}, {len(series)} puntos"
        )

        return series

    def train_model(self, series, model_type, **params):
        """
        Entrena un modelo específico.

        Args:
            series: Serie temporal para entrenamiento
            model_type: Tipo de modelo ('SARIMA', 'LSTM', 'GRU', 'XGBOOST')
            params: Parámetros adicionales para el modelo

        Returns:
            model: Modelo entrenado
        """
        logger.info(f"Entrenando modelo {model_type}")

        if len(series) < 12:
            logger.warning(
                f"La serie tiene solo {len(series)} puntos, se requieren al menos 12"
            )
            return None

        try:
            if model_type == "SARIMA":
                model = SarimaModel(**params)
                model.fit(series)
            elif model_type == "LSTM":
                model = LSTMModel(**params)
                model.fit(series)
            elif model_type == "GRU":
                model = GRUModel(**params)
                model.fit(series)
            elif model_type == "XGBOOST":
                # Parámetros simplificados para XGBoost
                lookback = params.get("lookback", 12)
                n_estimators = params.get("n_estimators", 100)
                learning_rate = params.get("learning_rate", 0.1)
                max_depth = params.get("max_depth", 5)

                model = XGBoostModel(
                    lookback=lookback,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                model.fit(series)
            else:
                raise ValueError(f"Tipo de modelo desconocido: {model_type}")

            return model
        except Exception as e:
            logger.error(f"Error al entrenar modelo {model_type}: {str(e)}")
            return None

    def generate_forecast(self, model, series, model_type, steps=6):
        """
        Genera pronósticos usando un modelo entrenado.

        Args:
            model: Modelo entrenado
            series: Serie temporal
            model_type: Tipo de modelo ('SARIMA', 'LSTM', 'GRU', 'XGBOOST')
            steps: Número de pasos a pronosticar

        Returns:
            pd.Series: Pronósticos validados
        """
        try:
            logger.info(
                f"Generando pronóstico con modelo {model_type} para {steps} pasos"
            )

            # Obtener última fecha de la serie
            last_date = series.index[-1]

            # Verificar que last_date sea un objeto datetime
            if not isinstance(last_date, pd.Timestamp) and not isinstance(
                last_date, datetime
            ):
                logger.info(
                    f"Convirtiendo fecha {last_date} de tipo {type(last_date)} a datetime"
                )
                # Si no es un objeto datetime, intentar convertirlo
                try:
                    if isinstance(last_date, str):
                        last_date = pd.to_datetime(last_date)
                    else:
                        # Si es otro tipo, convertir a string primero y luego a datetime
                        last_date = pd.to_datetime(str(last_date))
                except Exception as e:
                    logger.error(f"Error al convertir fecha: {str(e)}")
                    # Usar la fecha actual como fallback si hay error
                    last_date = pd.Timestamp.now()

            logger.info(f"Fecha base para pronóstico: {last_date}")

            # Generar fechas futuras (mensuales)
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=steps,
                freq="MS",
            )

            logger.info(
                f"Fechas futuras generadas: {[d.strftime('%Y-%m-%d') for d in future_dates]}"
            )
            
            # Validar integridad temporal
            validation_result = validate_temporal_integrity(series, pd.Series(index=future_dates), future_dates)
            if not validation_result['is_valid']:
                for error in validation_result['errors']:
                    logger.error(f"Error de integridad temporal: {error}")
            for warning in validation_result['warnings']:
                logger.warning(f"Advertencia temporal: {warning}")

            # Generar pronóstico según el tipo de modelo
            if model_type == "SARIMA":
                forecast_values = model.predict(steps=steps)
            elif model_type in ["LSTM", "GRU", "XGBOOST"]:
                forecast_values = model.predict(series, steps=steps)
            else:
                raise ValueError(f"Tipo de modelo desconocido: {model_type}")

            # Crear serie con los pronósticos
            forecast_series = pd.Series(forecast_values, index=future_dates)

            # Mostrar pronósticos originales para diagnóstico
            logger.info(f"Pronósticos originales de {model_type}:")
            for date, value in forecast_series.items():
                logger.info(f"  {date.strftime('%Y-%m-%d')}: {value:.4f}")

            # Validar y ajustar pronósticos si existe el método
            if hasattr(self, "validate_forecast"):
                # Obtener fecha de corte (fecha del último dato histórico)
                train_cutoff = series.index[-1] if len(series) > 0 else None
                validated_forecast = self.validate_forecast(
                    forecast_series, series, train_cutoff_date=train_cutoff
                )
                return validated_forecast
            else:
                return forecast_series

        except Exception as e:
            logger.error(
                f"Error al generar pronóstico con {model_type}: {str(e)}"
            )
            return pd.Series()

    def validate_forecast(self, forecast, series, train_cutoff_date=None):
        """
        Valida que los pronósticos estén dentro de rangos razonables basados SOLO en datos históricos 
        disponibles hasta la fecha de corte de entrenamiento.
        
        CORRECCIÓN PARA EVITAR DATA LEAKAGE:
        - Solo usa estadísticas de datos que estarían disponibles en el momento de hacer la predicción
        - Si se proporciona train_cutoff_date, solo usa datos hasta esa fecha
        - Evita usar información del conjunto de validación/prueba

        Args:
            forecast (pd.Series): Serie de pandas con los pronósticos a validar
            series (pd.Series): Serie de pandas con los datos históricos de referencia
            train_cutoff_date: Fecha límite de datos de entrenamiento (evita data leakage)

        Returns:
            pd.Series: Pronósticos validados/ajustados
        """
        try:
            # Verificar que tenemos datos suficientes
            if series.empty:
                logger.warning(
                    "Serie histórica vacía, no es posible validar pronósticos"
                )
                return forecast
            
            # Verificar que tenemos datos hasta la fecha de corte
            if train_cutoff_date is not None and len(recent_data) == 0:
                logger.warning(
                    f"No hay datos disponibles hasta la fecha de corte {train_cutoff_date}"
                )
                return forecast

            if forecast.empty:
                logger.warning("No hay pronósticos para validar")
                return forecast

            # CORRECCIÓN: Usar solo datos hasta la fecha de corte de entrenamiento
            if train_cutoff_date is not None:
                # Filtrar datos solo hasta la fecha de corte
                available_data = series[series.index <= train_cutoff_date]
                logger.info(f"Usando datos hasta {train_cutoff_date} para validación ({len(available_data)} puntos)")
            else:
                # Si no hay fecha de corte, usar toda la serie (comportamiento anterior)
                available_data = series
                logger.warning("No se proporcionó fecha de corte - usando toda la serie (puede causar data leakage)")
            
            # Usar datos recientes para calcular estadísticas (últimos 12 meses o todos si hay menos)
            recent_data = available_data[-12:] if len(available_data) > 12 else available_data

            # Imprimir estadísticas para diagnóstico
            logger.info(
                f"Validando pronósticos basados en {len(recent_data)} datos históricos recientes"
            )
            logger.info(f"Últimos valores disponibles para validación: {recent_data.tail(5).to_dict()}")

            # Calcular estadísticas
            recent_mean = recent_data.mean()
            recent_std = recent_data.std()
            recent_max = recent_data.max()
            recent_min = recent_data.min()

            logger.info(
                f"Estadísticas: Min={recent_min:.2f}, Max={recent_max:.2f}, Media={recent_mean:.2f}, Desv={recent_std:.2f}"
            )

            # Definir límites razonables
            # El límite superior es el máximo entre:
            # 1. 1.5 veces el valor máximo reciente
            # 2. La media + 2 desviaciones estándar
            upper_limit = max(recent_max * 1.5, recent_mean + 2 * recent_std)

            # El límite inferior es el máximo entre:
            # 1. Cero (para evitar valores negativos)
            # 2. La media - 2 desviaciones estándar (si es positivo)
            lower_limit = max(0, recent_mean - 2 * recent_std)

            logger.info(
                f"Límites calculados: Inferior={lower_limit:.2f}, Superior={upper_limit:.2f}"
            )

            # Crear copia de los pronósticos para no modificar el original
            adjusted_forecast = forecast.copy()

            # Variable para rastrear si se hicieron ajustes
            adjustments_made = False

            # Verificar y ajustar cada valor de pronóstico
            for date, value in forecast.items():
                # Verificar si el valor está fuera de los límites
                if value < lower_limit or value > upper_limit:
                    # Guardar valor original para logging
                    original_value = value

                    # Ajustar al límite más cercano
                    if value < lower_limit:
                        adjusted_forecast.at[date] = lower_limit
                    else:  # value > upper_limit
                        adjusted_forecast.at[date] = upper_limit

                    # Registrar el ajuste
                    logger.warning(
                        f"Ajuste: {date.strftime('%Y-%m-%d')} {original_value:.4f} → {adjusted_forecast.at[date]:.4f}"
                    )
                    adjustments_made = True
                else:
                    logger.debug(
                        f"Valor en rango: {date.strftime('%Y-%m-%d')} {value:.4f}"
                    )

            # Informar sobre resultados
            if adjustments_made:
                logger.info(
                    "Se ajustaron pronósticos para mantenerlos en rangos razonables"
                )
            else:
                logger.info(
                    "Todos los pronósticos están dentro de rangos razonables"
                )

            return adjusted_forecast

        except Exception as e:
            logger.error(f"Error al validar pronósticos: {str(e)}")
            # En caso de error, devolver los pronósticos originales
            return forecast

    def evaluate_model(self, train_series, test_series, model_type, **params):
        """
        Evalúa un modelo con datos de prueba.

        Args:
            train_series: Serie para entrenamiento
            test_series: Serie para evaluación
            model_type: Tipo de modelo
            params: Parámetros del modelo

        Returns:
            tuple: (métricas, modelo)
        """
        try:
            # Entrenar modelo
            model = self.train_model(train_series, model_type, **params)

            if model is None:
                return None, None

            # Generar pronóstico para el período de prueba
            steps = len(test_series)

            if model_type == "SARIMA":
                forecast = model.predict(steps=steps)
            elif model_type in [
                "LSTM",
                "GRU",
                "XGBOOST",
            ]:  # Añadido XGBOOST aquí
                forecast = model.predict(train_series, steps=steps)
            else:
                raise ValueError(f"Tipo de modelo desconocido: {model_type}")

            # Calcular métricas
            metrics = calculate_metrics(test_series.values, forecast)

            return metrics, model
        except Exception as e:
            logger.error(f"Error al evaluar modelo {model_type}: {str(e)}")
            return None, None

    def train_all_models(
        self, article, target="CANTIDADES", steps=6, debug=False
    ):
        """
        Entrena todos los modelos disponibles para un artículo.

        Args:
            article: Nombre del artículo
            target: Variable objetivo
            steps: Pasos a pronosticar
            debug: Modo depuración

        Returns:
            dict: Resultados de todos los modelos
        """
        # Preparar serie temporal
        series = self.prepare_time_series(article, target)

        if len(series) < 12:
            logger.warning(
                f"Serie insuficiente para {article}: {len(series)} puntos"
            )
            return {}

        # Dividir en entrenamiento y prueba usando método temporal estandarizado
        train_series, test_series, split_info = create_temporal_split(series, validation_split=0.2, min_train_samples=8)
        
        logger.info(f"División temporal ({split_info['method']}): entrenamiento={split_info['train_samples']}, prueba={split_info['val_samples']}")

        results = {}

        # Entrenar y evaluar SARIMA
        try:
            sarima_metrics, sarima_model = self.evaluate_model(
                train_series, test_series, "SARIMA"
            )

            if sarima_model is not None:
                # Entrenar con toda la serie para pronóstico
                full_sarima_model = self.train_model(series, "SARIMA")
                sarima_forecast = self.generate_forecast(
                    full_sarima_model, series, "SARIMA", steps=steps
                )

                results["SARIMA"] = {
                    "metrics": sarima_metrics,
                    "forecast": sarima_forecast,
                }

                # Guardar modelo para uso futuro
                self.models[f"{article}_{target}_SARIMA"] = full_sarima_model
        except Exception as e:
            logger.error(f"Error en proceso SARIMA: {str(e)}")

        # Entrenar y evaluar LSTM
        try:
            lstm_metrics, lstm_model = self.evaluate_model(
                train_series, test_series, "LSTM"
            )

            if lstm_model is not None:
                # Entrenar con toda la serie para pronóstico
                full_lstm_model = self.train_model(series, "LSTM")
                lstm_forecast = self.generate_forecast(
                    full_lstm_model, series, "LSTM", steps=steps
                )

                results["LSTM"] = {
                    "metrics": lstm_metrics,
                    "forecast": lstm_forecast,
                }

                # Guardar modelo para uso futuro
                self.models[f"{article}_{target}_LSTM"] = full_lstm_model
        except Exception as e:
            logger.error(f"Error en proceso LSTM: {str(e)}")

        # Entrenar y evaluar GRU
        try:
            gru_metrics, gru_model = self.evaluate_model(
                train_series, test_series, "GRU"
            )

            if gru_model is not None:
                # Entrenar con toda la serie para pronóstico
                full_gru_model = self.train_model(series, "GRU")
                gru_forecast = self.generate_forecast(
                    full_gru_model, series, "GRU", steps=steps
                )

                results["GRU"] = {
                    "metrics": gru_metrics,
                    "forecast": gru_forecast,
                }

                # Guardar modelo para uso futuro
                self.models[f"{article}_{target}_GRU"] = full_gru_model
        except Exception as e:
            logger.error(f"Error en proceso GRU: {str(e)}")

        try:
            # Comprobar si tenemos datos suficientes
            if len(series) >= 12:
                # Crear y entrenar modelo XGBoost directamente (sin evaluación)
                xgboost_model = XGBoostModel(lookback=12, n_estimators=100)
                xgboost_model.fit(series)

                # Generar pronóstico
                xgboost_forecast = xgboost_model.predict(series, steps=steps)

                # Crear serie temporal con fechas
                if (
                    hasattr(series.index, "freq")
                    and series.index.freq is not None
                ):
                    # Si la serie tiene una frecuencia definida, usar eso para generar fechas futuras
                    future_dates = pd.date_range(
                        start=series.index[-1] + pd.Timedelta(days=1),
                        periods=steps,
                        freq=series.index.freq,
                    )
                else:
                    # Si no, asumir frecuencia mensual
                    future_dates = pd.date_range(
                        start=series.index[-1] + pd.DateOffset(months=1),
                        periods=steps,
                        freq="MS",
                    )

                # Crear serie con pronósticos
                forecast_series = pd.Series(
                    xgboost_forecast, index=future_dates
                )

                # Calcular métricas de evaluación (usando últimos n datos como validación)
                val_size = min(6, len(series) // 3)
                if val_size > 0:
                    val_actual = series[-val_size:].values
                    val_pred = xgboost_model.predict(
                        series[:-val_size], steps=val_size
                    )
                    xgboost_metrics = calculate_metrics(val_actual, val_pred)
                else:
                    # Si no podemos calcular métricas, usar valores de relleno
                    xgboost_metrics = {
                        "MSE": 0.0,
                        "RMSE": 0.0,
                        "MAE": 0.0,
                        "MAPE": 0.0,
                        "R2": 0.0,
                    }

                # Guardar resultados
                results["XGBOOST"] = {
                    "metrics": xgboost_metrics,
                    "forecast": forecast_series,
                }

                # Guardar modelo para uso futuro
                self.models[f"{article}_{target}_XGBOOST"] = xgboost_model

                logger.info(
                    f"Modelo XGBOOST entrenado correctamente para {article}"
                )
            else:
                logger.warning(
                    f"Datos insuficientes para entrenar XGBOOST para {article}"
                )
        except Exception as e:
            logger.error(f"Error en proceso XGBOOST: {str(e)}")

        # Guardar resultados
        self.results[f"{article}_{target}"] = results

        return results

    def validate_forecast_deprecated(self, forecast, original_series):
        """
        Método obsoleto - usar validate_forecast con train_cutoff_date para evitar data leakage.
        
        NOTA: Este método se mantiene por compatibilidad pero puede causar data leakage
        ya que usa toda la serie histórica sin considerar fechas de corte.

        Args:
            forecast: Serie de pronósticos
            original_series: Serie histórica

        Returns:
            Series: Pronósticos validados/ajustados
        """
        logger.warning("Usando método validate_forecast obsoleto que puede causar data leakage")
        # Calcular estadísticas de los datos históricos (TODO: CORREGIR DATA LEAKAGE)
        hist_mean = original_series.mean()
        hist_std = original_series.std()
        hist_max = original_series.max()
        hist_min = original_series.min()
        
        logger.warning("ADVERTENCIA: Este método puede usar datos del futuro para validación")

        # Definir límites razonables
        upper_limit = max(hist_max * 1.5, hist_mean + 3 * hist_std)
        lower_limit = max(0, min(hist_min * 0.5, hist_mean - 3 * hist_std))

        # Verificar si algún pronóstico está fuera de límites
        corrected = False
        validated_forecast = forecast.copy()

        for i, value in enumerate(validated_forecast):
            if value > upper_limit or value < lower_limit:
                logger.warning(f"Pronóstico anómalo detectado: {value}")
                logger.warning(
                    f"Rango histórico: [{hist_min}, {hist_max}], Media: {hist_mean}"
                )

                # Ajustar valor
                if value > upper_limit:
                    validated_forecast[i] = upper_limit
                    corrected = True
                elif value < lower_limit:
                    validated_forecast[i] = lower_limit
                    corrected = True

        if corrected:
            logger.info(
                "Se han ajustado pronósticos anómalos al rango histórico"
            )

        return validated_forecast

    def get_best_model(self, article, target="CANTIDADES", metric="RMSE"):
        """
        Determina el mejor modelo según una métrica.

        Args:
            article: Nombre del artículo
            target: Variable objetivo
            metric: Métrica para comparación

        Returns:
            tuple: (mejor_modelo, pronóstico, métricas)
        """
        key = f"{article}_{target}"

        if key not in self.results:
            logger.warning(f"No hay resultados para {key}")
            return None, None, None

        results = self.results[key]
        best_model = None
        best_score = float("inf") if metric != "R2" else float("-inf")

        for model_name, model_results in results.items():
            if (
                "metrics" in model_results
                and metric in model_results["metrics"]
            ):
                score = model_results["metrics"][metric]

                # Para R2, mayor es mejor; para el resto, menor es mejor
                if (metric == "R2" and score > best_score) or (
                    metric != "R2" and score < best_score
                ):
                    best_score = score
                    best_model = model_name

        if best_model:
            return (
                best_model,
                results[best_model]["forecast"],
                results[best_model]["metrics"],
            )

        return None, None, None

    def optimize_xgboost(self, article, target="CANTIDADES", n_trials=50):
        """
        Optimiza los hiperparámetros del modelo XGBoost para un artículo específico.

        Args:
            article: Nombre del artículo
            target: Variable objetivo ('CANTIDADES' o 'PRODUCCIÓN')
            n_trials: Número de combinaciones de hiperparámetros a probar

        Returns:
            dict: Mejores hiperparámetros encontrados
        """
        from app.models.hyperparameter_tuner import XGBoostHyperparameterTuner

        logger.info(
            f"Iniciando optimización de XGBoost para {article} ({target})"
        )

        # Preparar serie temporal
        series = self.prepare_time_series(article, target)

        if len(series) < 24:
            logger.warning(
                f"Serie insuficiente para optimización: {len(series)} puntos, se recomiendan al menos 24"
            )
            return None

        # Inicializar optimizador
        tuner = XGBoostHyperparameterTuner(
            lookback=12,
            n_trials=n_trials,
            cv_splits=3,
            models_dir=MODELS_DIR,
            use_time_features=True,
        )

        try:
            # Realizar optimización
            best_params = tuner.optimize(
                series, article_name=f"{article}_{target}"
            )

            # Guardar mejor modelo optimizado
            optimized_model = XGBoostModel(
                lookback=12,
                n_estimators=best_params.get("n_estimators", 100),
                learning_rate=best_params.get("learning_rate", 0.1),
                max_depth=best_params.get("max_depth", 5),
            )

            # Entrenar con todos los datos
            optimized_model.fit(series)

            # Guardar el modelo optimizado
            self.models[
                f"{article}_{target}_XGBOOST_OPTIMIZED"
            ] = optimized_model

            # Obtener y mostrar importancia de características
            feature_importance = tuner.get_feature_importance()
            logger.info(
                f"Importancia de características para {article}: {feature_importance}"
            )

            return best_params

        except Exception as e:
            logger.error(f"Error durante la optimización de XGBoost: {str(e)}")
            return None

    def optimize_xgboost_with_progress(
        self, article, target="CANTIDADES", n_trials=50, progress_callback=None
    ):
        """
        Optimiza los hiperparámetros del modelo XGBoost con seguimiento de progreso.

        Args:
            article: Nombre del artículo
            target: Variable objetivo ('CANTIDADES' o 'PRODUCCIÓN')
            n_trials: Número de combinaciones de hiperparámetros a probar
            progress_callback: Función de callback para reportar progreso

        Returns:
            dict: Mejores hiperparámetros encontrados
        """
        from app.models.hyperparameter_tuner import XGBoostHyperparameterTuner

        logger.info(
            f"Iniciando optimización de XGBoost para {article} ({target})"
        )

        # Preparar serie temporal
        series = self.prepare_time_series(article, target)

        if len(series) < 24:
            logger.warning(
                f"Serie insuficiente para optimización: {len(series)} puntos, se recomiendan al menos 24"
            )
            if len(series) < 12:
                logger.error("Serie demasiado corta para optimización")
                if progress_callback:
                    # Reportar como completado en caso de error
                    progress_callback(100, n_trials, n_trials)
                return None

        # Log para verificar los datos
        logger.info(
            f"Estadísticas de serie para {article}: length={len(series)}, mean={series.mean():.2f}, std={series.std():.2f}"
        )

        try:
            # Inicializar optimizador
            tuner = XGBoostHyperparameterTuner(
                lookback=12,
                n_trials=n_trials,
                cv_splits=3,
                models_dir=MODELS_DIR,
                use_time_features=True,
            )

            # Log para verificar inicialización
            logger.info(f"Optimizador inicializado para {article}")

            # Realizar optimización con seguimiento de progreso
            best_params = tuner.optimize(
                series,
                article_name=f"{article}_{target}",
                progress_callback=progress_callback,
            )

            if best_params:
                logger.info(
                    f"Optimización exitosa para {article}. Mejores parámetros: {best_params}"
                )

                # Guardar mejor modelo optimizado
                optimized_model = XGBoostModel(
                    lookback=12,
                    n_estimators=best_params.get("n_estimators", 100),
                    learning_rate=best_params.get("learning_rate", 0.1),
                    max_depth=best_params.get("max_depth", 5),
                )

                # Entrenar con todos los datos
                optimized_model.fit(series)

                # Guardar el modelo optimizado
                self.models[
                    f"{article}_{target}_XGBOOST_OPTIMIZED"
                ] = optimized_model

                # Obtener y mostrar importancia de características
                feature_importance = tuner.get_feature_importance()
                logger.info(
                    f"Importancia de características para {article}: {feature_importance}"
                )

                return best_params
            else:
                logger.warning(
                    f"No se encontraron mejores parámetros para {article}"
                )
                return {}

        except Exception as e:
            logger.error(
                f"Error durante la optimización de XGBoost: {str(e)}",
                exc_info=True,
            )
            # Asegurar que reportamos finalización incluso en caso de error
            if progress_callback:
                try:
                    progress_callback(100, n_trials, n_trials)
                except:
                    pass
            return None

    def optimize_model_hyperparameters(
        self,
        article,
        model_type,
        target="CANTIDADES",
        n_trials=50,
        progress_callback=None,
    ):
        """
        Optimiza hiperparámetros para cualquier tipo de modelo.

        Args:
            article: Nombre del artículo
            model_type: Tipo de modelo ('SARIMA', 'LSTM', 'GRU', 'XGBOOST')
            target: Variable objetivo ('CANTIDADES' o 'PRODUCCIÓN')
            n_trials: Número de combinaciones de hiperparámetros a probar
            progress_callback: Función para reportar progreso

        Returns:
            dict: Mejores hiperparámetros encontrados
        """
        logger.info(
            f"Iniciando optimización de {model_type} para {article} ({target})"
        )

        # Preparar serie temporal
        series = self.prepare_time_series(article, target)

        if len(series) < 24:
            logger.warning(
                f"Serie insuficiente para optimización: {len(series)} puntos"
            )
            return None

        # Seleccionar optimizador según el tipo de modelo
        if model_type == "SARIMA":
            from app.models.optimization.sarima_tuner import (
                SarimaHyperparameterTuner,
            )

            tuner = SarimaHyperparameterTuner(
                n_trials=n_trials, models_dir=MODELS_DIR
            )
        elif model_type == "LSTM":
            from app.models.optimization.lstm_tuner import (
                LSTMHyperparameterTuner,
            )

            tuner = LSTMHyperparameterTuner(
                n_trials=n_trials, models_dir=MODELS_DIR
            )
        elif model_type == "GRU":
            from app.models.optimization.gru_tuner import GRUHyperparameterTuner

            tuner = GRUHyperparameterTuner(
                n_trials=n_trials, models_dir=MODELS_DIR
            )
        elif model_type == "XGBOOST":
            # Usar el optimizador existente
            return self.optimize_xgboost_with_progress(
                article, target, n_trials, progress_callback
            )
        else:
            logger.error(
                f"Tipo de modelo no soportado para optimización: {model_type}"
            )
            return None

        # Realizar optimización
        try:
            best_params = tuner.optimize(
                series,
                article_name=f"{article}_{target}",
                progress_callback=progress_callback,
            )

            if best_params:
                # Guardar parámetros óptimos para uso futuro
                key = f"{article}_{target}_{model_type}_OPTIMIZED_PARAMS"
                self.models[key] = best_params

                logger.info(
                    f"Optimización completada para {model_type}. Mejores parámetros: {best_params}"
                )
                return best_params
            else:
                logger.warning(
                    f"No se encontraron parámetros óptimos para {model_type}"
                )
                return {}
        except Exception as e:
            logger.error(
                f"Error durante la optimización de {model_type}: {str(e)}"
            )
            return None

    def train_xgboost_with_optimal_params(
        self, article, target="CANTIDADES", steps=6, use_cached_params=True
    ):
        """
        Entrena un modelo XGBoost con parámetros optimizados y genera pronósticos.

        Args:
            article: Nombre del artículo
            target: Variable objetivo
            steps: Pasos a pronosticar
            use_cached_params: Si se deben usar parámetros optimizados previamente

        Returns:
            pd.Series: Pronósticos generados
        """
        # Preparar serie temporal
        series = self.prepare_time_series(article, target)

        if len(series) < 12:
            logger.warning(
                f"Serie insuficiente para {article}: {len(series)} puntos"
            )
            return pd.Series()

        # Clave para modelos y parámetros
        key = f"{article}_{target}"

        # Verificar si tenemos parámetros optimizados en caché
        optimal_params = {}

        # 1. Si tenemos un modelo optimizado en memoria, usarlo directamente
        if f"{key}_XGBOOST_OPTIMIZED" in self.models:
            logger.info(
                f"Usando modelo XGBoost optimizado en memoria para {article}"
            )
            model = self.models[f"{key}_XGBOOST_OPTIMIZED"]
            return self.generate_forecast(model, series, "XGBOOST", steps=steps)

        # 2. Buscar en archivos guardados si hay parámetros optimizados
        if use_cached_params:
            try:
                import glob
                import os
                import pickle

                # Patrón de nombre de archivo para este artículo
                safe_name = (
                    article.replace(" ", "_")
                    .replace("/", "_")
                    .replace("\\", "_")
                )
                pattern = os.path.join(
                    MODELS_DIR, f"xgboost_optimized_{safe_name}_*.pkl"
                )

                # Buscar archivos que coincidan
                model_files = glob.glob(pattern)

                if model_files:
                    # Usar el más reciente
                    latest_file = max(model_files, key=os.path.getctime)
                    logger.info(
                        f"Encontrado modelo XGBoost optimizado: {latest_file}"
                    )

                    # Cargar parámetros
                    with open(latest_file, "rb") as f:
                        saved_data = pickle.load(f)
                        optimal_params = saved_data.get("params", {})
            except Exception as e:
                logger.warning(
                    f"Error al cargar parámetros optimizados: {str(e)}"
                )

        # 3. Si no hay parámetros optimizados, optimizar ahora
        if not optimal_params:
            logger.info(
                f"No se encontraron parámetros optimizados para {article}, optimizando..."
            )
            optimal_params = self.optimize_xgboost(article, target)

        # 4. Si tenemos parámetros optimizados, crear y entrenar modelo
        if optimal_params:
            logger.info(
                f"Entrenando XGBoost con parámetros optimizados: {optimal_params}"
            )
            model = XGBoostModel(
                lookback=12,
                n_estimators=optimal_params.get("n_estimators", 100),
                learning_rate=optimal_params.get("learning_rate", 0.1),
                max_depth=optimal_params.get("max_depth", 5),
            )

            # Entrenar modelo
            model.fit(series)

            # Guardar para uso futuro
            self.models[f"{key}_XGBOOST_OPTIMIZED"] = model

            # Generar pronóstico
            return self.generate_forecast(model, series, "XGBOOST", steps=steps)

        # 5. Si todo falló, usar parámetros por defecto
        logger.warning(
            f"Usando parámetros por defecto para XGBoost en {article}"
        )
        model = XGBoostModel()
        model.fit(series)
        return self.generate_forecast(model, series, "XGBOOST", steps=steps)

    def save_results(self, article, target="CANTIDADES"):
        """
        Guarda los resultados de pronósticos en un archivo JSON.

        Args:
            article: Nombre del artículo
            target: Variable objetivo

        Returns:
            str: Ruta al archivo guardado
        """
        key = f"{article}_{target}"

        if key not in self.results:
            logger.warning(f"No hay resultados para {key}")
            return None

        # Preparar datos para exportación
        export_data = {
            "article": article,
            "target": target,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "forecasts": {},
        }

        # Añadir pronósticos de cada modelo
        for model_name, model_results in self.results[key].items():
            if "forecast" in model_results:
                forecast = model_results["forecast"]
                dates = [d.strftime("%Y-%m-%d") for d in forecast.index]
                values = forecast.values.tolist()

                export_data["forecasts"][model_name] = {
                    "dates": dates,
                    "values": values,
                }

                if "metrics" in model_results:
                    export_data["forecasts"][model_name][
                        "metrics"
                    ] = model_results["metrics"]

        # Añadir información del mejor modelo
        best_model, _, best_metrics = self.get_best_model(article, target)
        if best_model:
            export_data["best_model"] = {
                "name": best_model,
                "metrics": best_metrics,
            }

        # Guardar a archivo
        filename = f"{article.replace(' ', '_')}_{target}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        file_path = os.path.join(RESULTS_DIR, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Resultados guardados en {file_path}")

        return file_path

    def save_model(self, article, target="CANTIDADES", model_type="best"):
        """
        Guarda un modelo entrenado para uso futuro.

        Args:
            article: Nombre del artículo
            target: Variable objetivo
            model_type: Tipo de modelo o 'best' para el mejor

        Returns:
            str: Ruta al archivo guardado
        """
        if model_type == "best":
            best_model, _, _ = self.get_best_model(article, target)
            if best_model:
                model_type = best_model
            else:
                logger.warning(
                    f"No se encontró mejor modelo para {article}_{target}"
                )
                return None

        model_key = f"{article}_{target}_{model_type}"

        if model_key not in self.models:
            logger.warning(f"No hay modelo guardado para {model_key}")
            return None

        # Guardar modelo
        filename = f"{article.replace(' ', '_')}_{target}_{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
        file_path = os.path.join(MODELS_DIR, filename)

        with open(file_path, "wb") as f:
            pickle.dump(self.models[model_key], f)

        logger.info(f"Modelo guardado en {file_path}")

        return file_path

    def plot_forecasts(self, article, target="CANTIDADES", output_path=None):
        """
        Genera gráfico con pronósticos para un artículo.

        Args:
            article: Nombre del artículo
            target: Variable objetivo
            output_path: Ruta para guardar el gráfico

        Returns:
            matplotlib.figure o None
        """
        key = f"{article}_{target}"

        if key not in self.results:
            logger.warning(f"No hay resultados para {key}")
            return None

        # Preparar serie original
        series = self.prepare_time_series(article, target)

        if len(series) == 0:
            logger.warning(f"No hay datos para {article}")
            return None

        # Crear figura
        plt.figure(figsize=(12, 6))

        # Graficar serie original
        plt.plot(
            series.index, series.values, label="Datos históricos", linewidth=2
        )

        # Graficar pronósticos
        colors = ["#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        i = 0

        for model_name, model_results in self.results[key].items():
            if "forecast" in model_results:
                forecast = model_results["forecast"]
                plt.plot(
                    forecast.index,
                    forecast.values,
                    label=f"Pronóstico {model_name}",
                    linestyle="--",
                    linewidth=2,
                    color=colors[i % len(colors)],
                )
                i += 1

        # Configurar gráfico
        plt.title(f"Pronóstico de {target} para {article}")
        plt.xlabel("Fecha")
        plt.ylabel(target)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Guardar si se especifica ruta
        if output_path:
            plt.savefig(output_path, dpi=100)
            logger.info(f"Gráfico guardado en {output_path}")

        return plt.gcf()

    def load_leadtimes(self, file_path):
        """
        Carga datos de tiempos de entrega desde un archivo Excel.

        Args:
            file_path: Ruta al archivo Excel

        Returns:
            dict: Diccionario con los tiempos de entrega
        """
        return self.leadtime_handler.load_from_excel(file_path)

    def get_leadtime(self, article):
        """
        Obtiene el tiempo de entrega para un artículo.

        Args:
            article: Nombre o código del artículo

        Returns:
            int: Tiempo de entrega en días
        """
        return self.leadtime_handler.get_leadtime(article)

    def update_leadtime(self, article, leadtime):
        """
        Actualiza el tiempo de entrega para un artículo.

        Args:
            article: Nombre o código del artículo
            leadtime: Nuevo tiempo de entrega en días

        Returns:
            bool: True si se actualizó correctamente
        """
        return self.leadtime_handler.update_leadtime(article, leadtime)

    def export_leadtimes(self, file_path):
        """
        Exporta los tiempos de entrega a un archivo Excel.

        Args:
            file_path: Ruta para guardar el archivo Excel

        Returns:
            bool: True si se exportó correctamente
        """
        return self.leadtime_handler.export_to_excel(file_path)

    def calculate_safety_stock(
        self,
        article,
        target="CANTIDADES",
        method="basic",
        service_level=0.95,
        review_period=None,
        forecast_horizon=6,
        use_forecasts=True,
    ):
        """
        Calcula el stock de seguridad para un artículo.

        Args:
            article: Nombre del artículo
            target: Variable objetivo ('CANTIDADES' o 'PRODUCCIÓN')
            method: Método de cálculo ('basic', 'leadtime_var', 'review', 'insufficient', 'forecast')
            service_level: Nivel de servicio deseado (0-1)
            review_period: Período de revisión del inventario (días)
            forecast_horizon: Horizonte de pronóstico en meses (6 o 12 típicamente)
            use_forecasts: Si se deben usar pronósticos para el cálculo

        Returns:
            dict: Resultados del cálculo de stock de seguridad
        """
        try:
            # Obtener serie temporal del artículo
            series = self.prepare_time_series(article, target)

            # Si no hay datos suficientes, usar método para datos insuficientes
            if len(series) < 12:
                method = "insufficient"
                use_forecasts = (
                    False  # No usar pronósticos si no hay datos suficientes
                )
                logger.warning(
                    f"Datos insuficientes para {article}, usando método alternativo"
                )

            # Obtener tiempo de entrega en días
            leadtime_days = self.get_leadtime(article)

            # Convertir tiempo de entrega de días a períodos
            # Asumimos que los datos son mensuales, por lo que dividimos los días por 30
            leadtime_periods = leadtime_days / 30.0

            # Analizar patrón de demanda
            demand_analysis = (
                self.safety_stock_calculator.analyze_demand_pattern(series)
            )

            # Recomendar nivel de servicio si no se especifica
            if service_level is None:
                service_level = (
                    self.safety_stock_calculator.recommend_service_level(
                        demand_analysis, item_importance="medium"
                    )
                )

            # Variables para almacenar los resultados
            safety_stock = 0
            safety_stocks_by_month = None
            best_model_info = None
            forecasts = None

            # Si debemos usar pronósticos, intentamos generarlos
            if use_forecasts and method != "insufficient":
                # Generar pronósticos usando el mejor modelo
                try:
                    # Entrenar todos los modelos y obtener el mejor
                    logger.info(
                        f"Generando pronósticos para calcular stock de seguridad de {article}"
                    )
                    results = self.train_all_models(
                        article, target=target, steps=forecast_horizon
                    )

                    if results:
                        # Obtener el mejor modelo
                        best_model, forecast, metrics = self.get_best_model(
                            article, target
                        )

                        if (
                            best_model
                            and forecast is not None
                            and len(forecast) > 0
                        ):
                            # Guardar info del mejor modelo
                            best_model_info = {
                                "name": best_model,
                                "metrics": metrics,
                            }

                            # Guardar los pronósticos
                            forecasts = forecast.values

                            logger.info(
                                f"Usando modelo {best_model} para calcular stock de seguridad"
                            )

                            # Usar método específico para pronósticos
                            if method == "forecast" or method == "basic":
                                # Obtener error de pronóstico como medida de variabilidad
                                forecast_error = metrics.get(
                                    "RMSE", series.std()
                                )

                                # Calcular stock de seguridad por mes
                                safety_stocks = self.safety_stock_calculator.calculate_with_forecast(
                                    forecasts,
                                    forecast_error,
                                    leadtime_periods,
                                    service_level,
                                )

                                # Guardar stock de seguridad promedio y por mes
                                safety_stock = sum(safety_stocks) / len(
                                    safety_stocks
                                )
                                safety_stocks_by_month = {}

                                # Asociar cada stock de seguridad con su fecha
                                for i, ss in enumerate(safety_stocks):
                                    date = forecast.index[i]
                                    date_str = (
                                        date.strftime("%Y-%m-%d")
                                        if hasattr(date, "strftime")
                                        else str(date)
                                    )
                                    safety_stocks_by_month[date_str] = round(
                                        ss, 2
                                    )

                                # Usar método de pronóstico
                                method = "forecast"
                            else:
                                # Si el método no es 'forecast', seguimos con el flujo normal
                                # pero incluimos los pronósticos en el resultado
                                pass
                except Exception as e:
                    logger.error(
                        f"Error al generar pronósticos para stock de seguridad: {str(e)}"
                    )
                    # Si hay error, continuamos con el cálculo normal

            # Si no estamos usando el método de pronóstico o falló, usamos los métodos tradicionales
            if method != "forecast":
                if method == "basic":
                    # Calcular stock de seguridad con método básico
                    avg_demand = series.mean()
                    std_dev = series.std()
                    safety_stock = self.safety_stock_calculator.calculate_basic(
                        avg_demand, std_dev, leadtime_periods, service_level
                    )

                elif method == "leadtime_var":
                    # Calcular con variabilidad en el tiempo de entrega
                    # Asumimos una variabilidad del 20% del tiempo de entrega
                    avg_demand = series.mean()
                    std_dev_demand = series.std()
                    avg_leadtime = leadtime_periods
                    std_dev_leadtime = leadtime_periods * 0.2

                    safety_stock = self.safety_stock_calculator.calculate_with_leadtime_variability(
                        avg_demand,
                        std_dev_demand,
                        avg_leadtime,
                        std_dev_leadtime,
                        service_level,
                    )

                elif method == "review":
                    # Calcular considerando período de revisión
                    std_dev = series.std()

                    # Si no se especifica período de revisión, asumir 1 mes
                    if review_period is None:
                        review_period = 1.0
                    else:
                        # Convertir días a meses
                        review_period = review_period / 30.0

                    safety_stock = self.safety_stock_calculator.calculate_with_review_period(
                        std_dev, leadtime_periods, review_period, service_level
                    )

                else:  # 'insufficient' u otro método no reconocido
                    # Usar método para datos insuficientes
                    safety_stock = self.safety_stock_calculator.calculate_for_insufficient_data(
                        series.values,
                        leadtime_periods,
                        min_safety_factor=0.3,
                        service_level=service_level,
                    )

            # Cálculo del punto de reorden
            daily_demand = (
                series.mean() / 30
            )  # Convertir demanda mensual a diaria
            leadtime_demand = daily_demand * leadtime_days
            reorder_point = leadtime_demand + safety_stock

            # Preparar resultados
            results = {
                "article": article,
                "target": target,
                "method": method,
                "service_level": service_level,
                "leadtime_days": leadtime_days,
                "leadtime_periods": leadtime_periods,
                "safety_stock": round(safety_stock, 2),
                "reorder_point": round(
                    reorder_point, 2
                ),  # Añadido punto de reorden
                "demand_analysis": demand_analysis,
                "data_points": len(series),
                "avg_demand": series.mean() if len(series) > 0 else 0,
                "daily_demand": daily_demand,  # Añadida demanda diaria para referencia
                "std_dev": series.std() if len(series) > 1 else 0,
            }

            # Añadir período de revisión si se especificó
            if review_period is not None:
                results["review_period"] = review_period

            # Añadir información de pronósticos si está disponible
            if forecasts is not None:
                results["forecasts"] = forecasts.tolist()

            # Añadir información del mejor modelo si está disponible
            if best_model_info is not None:
                results["best_model"] = best_model_info

            # Añadir stock de seguridad por mes si está disponible
            if safety_stocks_by_month is not None:
                results["safety_stocks_by_month"] = safety_stocks_by_month

            results = self._convert_numpy_types(results)

            return results

        except Exception as e:
            logger.error(
                f"Error al calcular stock de seguridad para {article}: {str(e)}"
            )
            return {
                "article": article,
                "error": str(e),
                "safety_stock": 0,
                "reorder_point": 0,
            }

    def calculate_all_safety_stocks(
        self,
        target="CANTIDADES",
        method="basic",
        service_level=0.95,
        forecast_horizon=6,
        use_forecasts=True,
    ):
        """
        Calcula el stock de seguridad para todos los artículos disponibles.

        Args:
            target: Variable objetivo ('CANTIDADES' o 'PRODUCCIÓN')
            method: Método de cálculo ('basic', 'leadtime_var', 'review', 'insufficient', 'forecast')
            service_level: Nivel de servicio deseado (0-1)
            forecast_horizon: Horizonte de pronóstico en meses
            use_forecasts: Si se deben usar pronósticos para el cálculo

        Returns:
            dict: Resultados del cálculo de stock de seguridad por artículo
        """
        results = {}

        if self.articles is None or len(self.articles) == 0:
            logger.warning(
                "No hay artículos disponibles para calcular stock de seguridad"
            )
            return results

        for article in self.articles:
            results[article] = self.calculate_safety_stock(
                article,
                target,
                method,
                service_level,
                forecast_horizon=forecast_horizon,
                use_forecasts=use_forecasts,
            )

        return results

    def export_safety_stock_results(self, results, file_path):
        """
        Exporta los resultados de stock de seguridad a un archivo Excel.

        Args:
            results: Diccionario con resultados por artículo
            file_path: Ruta para guardar el archivo Excel

        Returns:
            bool: True si se exportó correctamente
        """
        try:
            # Preparar datos para exportar
            data = []
            for article, result in results.items():
                row = {
                    "Artículo": article,
                    "Stock de Seguridad": result.get("safety_stock", 0),
                    "Nivel de Servicio": result.get("service_level", 0) * 100,
                    "Tiempo de Entrega (días)": result.get("leadtime_days", 0),
                    "Método de Cálculo": result.get("method", "unknown"),
                    "Puntos de Datos": result.get("data_points", 0),
                    "Demanda Promedio": result.get("avg_demand", 0),
                    "Desviación Estándar": result.get("std_dev", 0),
                }
                data.append(row)

            # Crear DataFrame y exportar a Excel
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False)
            logger.info(
                f"Resultados de stock de seguridad exportados a {file_path}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error al exportar resultados de stock de seguridad: {str(e)}"
            )
            return False

    def _convert_numpy_types(self, obj):
        """
        Convierte tipos de NumPy a tipos nativos de Python para serialización JSON.
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        else:
            return obj