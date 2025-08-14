import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Clase para cargar y preprocesar datos de inventario."""

    @staticmethod
    def load_csv(file_path, separator=";", encoding="utf-8"):
        """
        Carga un archivo CSV.

        Args:
            file_path: Ruta al archivo CSV
            separator: Separador de columnas (por defecto ';')
            encoding: Codificación del archivo

        Returns:
            DataFrame: Datos cargados
        """
        try:
            df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            logger.info(f"Archivo cargado: {file_path}, {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"Error al cargar archivo {file_path}: {str(e)}")
            raise

    @staticmethod
    def clean_data(df):
        """
        Limpia y preprocesa los datos.

        Args:
            df: DataFrame con datos brutos

        Returns:
            DataFrame: Datos limpios
        """
        # Copia para no modificar el original
        clean_df = df.copy()

        # Renombrar columnas si es necesario
        if clean_df.columns[0] != "Fecha":
            clean_df = clean_df.rename(columns={clean_df.columns[0]: "Fecha"})

        # Eliminar filas con valores nulos en columnas clave
        key_columns = ["Fecha", "CENTRO DE COSTO", "ARTÍCULO"]
        clean_df = clean_df.dropna(subset=key_columns)

        # Limpiar espacios en texto
        for col in clean_df.columns:
            if clean_df[col].dtype == "object":
                clean_df[col] = clean_df[col].str.strip()

        # Limpiar valores numéricos
        numeric_cols = ["CANTIDADES", "PRODUCCIÓN"]
        for col in numeric_cols:
            if col in clean_df.columns:
                clean_df[col] = DataLoader._clean_numeric_column(clean_df[col])

        # Convertir fechas
        clean_df["fecha_std"] = clean_df["Fecha"].apply(
            DataLoader._standardize_date
        )

        # Asegurarse de que tenemos fechas válidas
        fecha_na_count = clean_df["fecha_std"].isna().sum()
        if fecha_na_count > 0:
            logger.warning(
                f"Se encontraron {fecha_na_count} fechas inválidas en los datos"
            )

        # Verificar si tenemos al menos una fecha válida
        if clean_df["fecha_std"].notna().any():
            # Asegurarse de que todas las fechas son datetime
            if not pd.api.types.is_datetime64_dtype(clean_df["fecha_std"]):
                logger.warning("Convirtiendo valores de fecha a datetime")
                clean_df["fecha_std"] = pd.to_datetime(
                    clean_df["fecha_std"], errors="coerce"
                )
        else:
            logger.warning("No se encontraron fechas válidas en los datos")

        return clean_df

    @staticmethod
    def _clean_numeric_column(column):
        """
        Limpia una columna numérica considerando el formato español/latino
        (puntos como separadores de miles, comas como separadores decimales).

        Si los datos ya se cargaron con pd.read_csv(..., decimal=',', thousands='.'),
        este método solo necesita manejar casos donde se procesen valores directamente.

        Args:
            column: Serie o valores a limpiar

        Returns:
            Serie con valores numéricos correctamente interpretados
        """
        # Si ya es un valor numérico, simplemente lo devolvemos
        if pd.api.types.is_numeric_dtype(column):
            return column

        # Si es una Serie de texto, procesamos
        if isinstance(column, pd.Series):
            # Verificar valores antes de procesar (para diagnóstico)
            sample = column.head(3).tolist()
            logger.debug(f"Muestra antes de limpieza: {sample}")

            # Primero eliminar puntos (separadores de miles), luego convertir comas a puntos (decimales)
            cleaned = column.str.strip()
            cleaned = cleaned.str.replace(".", "", regex=False)
            cleaned = cleaned.str.replace(",", ".", regex=False)

            # Verificar después del procesamiento
            sample_after = cleaned.head(3).tolist()
            logger.debug(f"Muestra después de limpieza: {sample_after}")
        else:
            # Si no es una Serie, convertirlo primero
            cleaned = pd.Series([str(x).strip() for x in column])
            cleaned = cleaned.str.replace(".", "", regex=False)
            cleaned = cleaned.str.replace(",", ".", regex=False)

        # Convertir a tipo numérico
        return pd.to_numeric(cleaned, errors="coerce")

    @staticmethod
    def _standardize_date(date_str):
        """Convierte fecha de formato 'mes-año' a 'año-mes-01'."""
        # Si ya es un datetime, devolverlo directamente
        if isinstance(date_str, (pd.Timestamp, datetime)):
            return date_str

        # Si no es un string, devolver NaT
        if not isinstance(date_str, str):
            return pd.NaT

        # Normalizar el string
        date_str = date_str.strip().lower()

        # Dividir por el guión
        parts = date_str.split("-")
        if len(parts) != 2:
            return pd.NaT

        # Mapa de nombres de meses a números
        month_map = {
            "ene": "01",
            "feb": "02",
            "mar": "03",
            "abr": "04",
            "may": "05",
            "jun": "06",
            "jul": "07",
            "ago": "08",
            "sep": "09",
            "sept": "09",
            "oct": "10",
            "nov": "11",
            "dic": "12",
            # Añadir abreviaturas en inglés por si acaso
            "jan": "01",
            "feb": "02",
            "mar": "03",
            "apr": "04",
            "may": "05",
            "jun": "06",
            "jul": "07",
            "aug": "08",
            "sep": "09",
            "sept": "09",
            "oct": "10",
            "nov": "11",
            "dec": "12",
        }

        # Obtener el mes del mapa
        month_str = parts[0].lower()
        month = month_map.get(month_str)

        # Si no se encuentra en el mapa, intentar interpretar como número
        if not month and month_str.isdigit() and 1 <= int(month_str) <= 12:
            month = f"{int(month_str):02d}"

        # Si aún no tenemos un mes válido, devolver NaT
        if not month:
            return pd.NaT

        # Normalizar el año
        year_str = parts[1].strip()
        if len(year_str) == 2:
            year = "20" + year_str  # Asumimos años 2000+
        elif len(year_str) == 4 and year_str.isdigit():
            year = year_str
        else:
            return pd.NaT

        # Intentar crear el datetime
        try:
            return pd.to_datetime(f"{year}-{month}-01")
        except Exception:
            return pd.NaT