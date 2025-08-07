import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class SafetyStockCalculator:
    """
    Clase para calcular el stock de seguridad según diferentes métodos.

    Esta clase proporciona múltiples métodos para calcular el stock de seguridad
    considerando diferentes factores como tiempo de entrega, desviación estándar
    de la demanda, nivel de servicio, etc.
    """

    def __init__(self):
        """Inicializa el calculador de stock de seguridad."""
        # Mapa de niveles de servicio a factores Z (unilateral)
        self.service_level_factors = {
            0.50: 0.00,  # 50% nivel de servicio
            0.75: 0.67,  # 75% nivel de servicio
            0.80: 0.84,  # 80% nivel de servicio
            0.85: 1.04,  # 85% nivel de servicio
            0.90: 1.28,  # 90% nivel de servicio
            0.95: 1.65,  # 95% nivel de servicio
            0.96: 1.75,  # 96% nivel de servicio
            0.97: 1.88,  # 97% nivel de servicio
            0.98: 2.05,  # 98% nivel de servicio
            0.99: 2.33,  # 99% nivel de servicio
            0.995: 2.58,  # 99.5% nivel de servicio
            0.999: 3.09,  # 99.9% nivel de servicio
        }

    def get_z_factor(self, service_level):
        """
        Obtiene el factor Z para un nivel de servicio dado.

        Args:
            service_level: Nivel de servicio deseado (0-1)

        Returns:
            float: Factor Z correspondiente
        """
        # Si el nivel de servicio está en nuestro mapa, usarlo directamente
        if service_level in self.service_level_factors:
            return self.service_level_factors[service_level]

        # Si no, calcular el valor exacto usando la distribución normal
        try:
            # Asegurar que el nivel de servicio esté en el rango válido
            service_level = max(0.5, min(0.9999, service_level))
            return stats.norm.ppf(service_level)
        except:
            # En caso de error, usar un valor conservador
            return 1.65  # ~95% nivel de servicio

    def calculate_basic(
        self, avg_demand, std_dev, leadtime, service_level=0.95
    ):
        """
        Cálculo básico: Z × σ × √L

        Args:
            avg_demand: Demanda promedio por período
            std_dev: Desviación estándar de la demanda
            leadtime: Tiempo de entrega en períodos
            service_level: Nivel de servicio deseado (0-1)

        Returns:
            float: Stock de seguridad calculado
        """
        z = self.get_z_factor(service_level)
        safety_stock = z * std_dev * np.sqrt(leadtime)

        # Nunca devolver un valor negativo
        return max(0, safety_stock)

    def calculate_with_forecast(
        self, forecasts, forecast_error, leadtime, service_level=0.95
    ):
        """
        Calcula el stock de seguridad para cada período utilizando pronósticos.

        Args:
            forecasts: Lista o array con valores pronosticados
            forecast_error: Error de pronóstico (RMSE o similar)
            leadtime: Tiempo de entrega en períodos
            service_level: Nivel de servicio deseado (0-1)

        Returns:
            list: Stock de seguridad para cada período del horizonte
        """
        z = self.get_z_factor(service_level)
        safety_stocks = []

        # Convertir a array de numpy si es necesario
        if not isinstance(forecasts, np.ndarray):
            forecasts = np.array(forecasts)

        # Calcular stock de seguridad para cada período
        for forecast in forecasts:
            # Usamos el error de pronóstico como medida de variabilidad
            # y lo multiplicamos por un factor relacionado con el valor pronosticado
            # para reflejar que la variabilidad suele aumentar con la magnitud

            # Calcular la desviación estándar efectiva
            # Si el pronóstico es muy bajo, usamos un mínimo para evitar stocks de seguridad demasiado bajos
            effective_std = max(forecast_error, forecast * 0.1)

            # Calcular stock de seguridad
            safety_stock = z * effective_std * np.sqrt(leadtime)

            # Nunca devolver un valor negativo
            safety_stocks.append(max(0, safety_stock))

        return safety_stocks

    def calculate_with_leadtime_variability(
        self,
        avg_demand,
        std_dev_demand,
        avg_leadtime,
        std_dev_leadtime,
        service_level=0.95,
    ):
        """
        Cálculo con variabilidad en el tiempo de entrega:
        Z × √(L × σ_d² + d² × σ_L²)

        Args:
            avg_demand: Demanda promedio por período
            std_dev_demand: Desviación estándar de la demanda
            avg_leadtime: Tiempo de entrega promedio
            std_dev_leadtime: Desviación estándar del tiempo de entrega
            service_level: Nivel de servicio deseado (0-1)

        Returns:
            float: Stock de seguridad calculado
        """
        z = self.get_z_factor(service_level)

        # Calcular término bajo la raíz
        term = avg_leadtime * (std_dev_demand**2) + (avg_demand**2) * (
            std_dev_leadtime**2
        )

        # Calcular stock de seguridad
        safety_stock = z * np.sqrt(max(0, term))

        return safety_stock

    def calculate_with_review_period(
        self, std_dev, leadtime, review_period, service_level=0.95
    ):
        """
        Cálculo considerando período de revisión: Z × σ × √(L + R)

        Args:
            std_dev: Desviación estándar de la demanda
            leadtime: Tiempo de entrega en períodos
            review_period: Período de revisión del inventario
            service_level: Nivel de servicio deseado (0-1)

        Returns:
            float: Stock de seguridad calculado
        """
        z = self.get_z_factor(service_level)
        safety_stock = z * std_dev * np.sqrt(leadtime + review_period)

        return max(0, safety_stock)

    def calculate_for_insufficient_data(
        self,
        limited_demand_data,
        leadtime,
        min_safety_factor=0.2,
        service_level=0.95,
    ):
        """
        Cálculo para artículos con datos insuficientes.

        Args:
            limited_demand_data: Lista o array con datos limitados de demanda
            leadtime: Tiempo de entrega en períodos
            min_safety_factor: Factor mínimo de seguridad (porcentaje de la demanda promedio)
            service_level: Nivel de servicio deseado (0-1)

        Returns:
            float: Stock de seguridad calculado
        """
        # Convertir a array de numpy si es necesario
        if not isinstance(limited_demand_data, np.ndarray):
            limited_demand_data = np.array(limited_demand_data)

        # Filtrar valores no válidos
        valid_data = limited_demand_data[np.isfinite(limited_demand_data)]

        # Si no hay datos válidos suficientes
        if len(valid_data) < 3:
            logger.warning(
                "Datos insuficientes para cálculo estadístico de stock de seguridad"
            )
            # Usar método alternativo basado en el máximo valor conocido
            if len(valid_data) > 0:
                max_demand = np.max(valid_data)
                avg_demand = np.mean(valid_data)
                safety_stock = max(
                    max_demand * 0.5, avg_demand * min_safety_factor
                ) * np.sqrt(leadtime)
            else:
                logger.error("No hay datos de demanda disponibles")
                safety_stock = 0
        else:
            # Calcular con el método básico
            avg_demand = np.mean(valid_data)
            std_dev = np.std(
                valid_data, ddof=1
            )  # ddof=1 para estimación insesgada

            # Usar un factor de corrección para muestras pequeñas
            correction = max(1.0, 1.0 + 2.0 / len(valid_data))
            std_dev = std_dev * correction

            # Calcular stock de seguridad
            z = self.get_z_factor(service_level)
            safety_stock = z * std_dev * np.sqrt(leadtime)

            # Aplicar mínimo basado en la demanda promedio
            min_safety = avg_demand * min_safety_factor * np.sqrt(leadtime)
            safety_stock = max(safety_stock, min_safety)

        return max(0, safety_stock)

    def analyze_demand_pattern(self, demand_data):
        """
        Analiza el patrón de demanda para determinar características
        como estacionalidad, tendencia, y variabilidad.

        Args:
            demand_data: Serie temporal de demanda

        Returns:
            dict: Diccionario con análisis de la demanda
        """
        analysis = {}

        # Convertir a array de numpy si es necesario
        if isinstance(demand_data, pd.Series):
            values = demand_data.values
            has_index = True
        else:
            values = np.array(demand_data)
            has_index = False

        # Filtrar valores no válidos
        valid_data = values[np.isfinite(values)]

        # Si no hay suficientes datos, devolver análisis limitado
        if len(valid_data) < 6:
            analysis["sufficient_data"] = False
            analysis["count"] = len(valid_data)
            if len(valid_data) > 0:
                analysis["mean"] = np.mean(valid_data)
                analysis["max"] = np.max(valid_data)
                analysis["min"] = np.min(valid_data)
            return analysis

        # Estadísticas básicas
        analysis["sufficient_data"] = True
        analysis["count"] = len(valid_data)
        analysis["mean"] = np.mean(valid_data)
        analysis["median"] = np.median(valid_data)
        analysis["std_dev"] = np.std(valid_data, ddof=1)
        analysis["cv"] = (
            analysis["std_dev"] / analysis["mean"]
            if analysis["mean"] > 0
            else 0
        )
        analysis["max"] = np.max(valid_data)
        analysis["min"] = np.min(valid_data)

        # Detectar valores atípicos (outliers)
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = valid_data[
            (valid_data < lower_bound) | (valid_data > upper_bound)
        ]
        analysis["outliers_count"] = len(outliers)
        analysis["outliers_percentage"] = len(outliers) / len(valid_data) * 100

        # Calcular tendencia (pendiente de regresión lineal)
        if len(valid_data) >= 12 and has_index:
            try:
                x = np.arange(len(valid_data))
                slope, _, _, _, _ = stats.linregress(x, valid_data)
                analysis["trend_slope"] = slope
                analysis["trend_significant"] = (
                    abs(slope) > 0.05 * analysis["mean"]
                )
            except:
                analysis["trend_slope"] = 0
                analysis["trend_significant"] = False

        # Coeficiente de variación para determinar tipo de demanda
        if analysis["cv"] < 0.2:
            analysis["demand_type"] = "Estable"
        elif analysis["cv"] < 0.5:
            analysis["demand_type"] = "Irregular"
        else:
            analysis["demand_type"] = "Errática"

        # Calcular el porcentaje de períodos con cero demanda
        analysis["zero_demand_percentage"] = (
            np.sum(valid_data == 0) / len(valid_data) * 100
        )
        if analysis["zero_demand_percentage"] > 30:
            analysis["demand_type"] += " con muchos ceros"

        return analysis

    def recommend_service_level(
        self, demand_analysis, item_importance="medium"
    ):
        """
        Recomienda un nivel de servicio basado en el análisis de la demanda
        y la importancia del artículo.

        Args:
            demand_analysis: Diccionario con análisis de la demanda
            item_importance: Importancia del artículo ('low', 'medium', 'high', 'critical')

        Returns:
            float: Nivel de servicio recomendado
        """
        # Factores base según importancia
        importance_factor = {
            "low": 0.85,
            "medium": 0.90,
            "high": 0.95,
            "critical": 0.98,
        }

        base_service_level = importance_factor.get(item_importance, 0.90)

        # Ajustar según el tipo de demanda
        if demand_analysis.get("sufficient_data", False):
            demand_type = demand_analysis.get("demand_type", "Estable")

            # Incrementar para demanda errática
            if "Errática" in demand_type:
                base_service_level = min(0.99, base_service_level + 0.03)

            # Reducir para demanda con muchos ceros
            if "ceros" in demand_type:
                base_service_level = max(0.80, base_service_level - 0.05)

            # Incrementar si hay tendencia significativa
            if demand_analysis.get("trend_significant", False):
                # Si la tendencia es positiva, aumentar nivel de servicio
                if demand_analysis.get("trend_slope", 0) > 0:
                    base_service_level = min(0.99, base_service_level + 0.02)

        return base_service_level

    def calculate_reorder_point(
        self,
        avg_demand,
        std_dev,
        leadtime,
        service_level=0.95,
        time_unit="monthly",
    ):
        """
        Calcula el punto de reorden (ROP) basado en la demanda durante el tiempo de entrega
        más el stock de seguridad.

        Args:
            avg_demand: Demanda promedio por período
            std_dev: Desviación estándar de la demanda
            leadtime: Tiempo de entrega en días
            service_level: Nivel de servicio deseado (0-1)
            time_unit: Unidad de tiempo de la demanda ('monthly', 'weekly', 'daily')

        Returns:
            float: Punto de reorden calculado
        """
        # Convertir la demanda a unidades diarias si es necesario
        daily_demand = avg_demand
        if time_unit == "monthly":
            daily_demand = avg_demand / 30.0
        elif time_unit == "weekly":
            daily_demand = avg_demand / 7.0

        # Calcular demanda durante el tiempo de entrega
        leadtime_demand = daily_demand * leadtime

        # Calcular stock de seguridad
        safety_stock = self.calculate_basic(
            avg_demand, std_dev, leadtime / 30.0, service_level
        )

        # Calcular punto de reorden
        reorder_point = leadtime_demand + safety_stock

        return reorder_point
