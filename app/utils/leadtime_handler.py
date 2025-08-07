import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


class LeadTimeHandler:
    """Clase para manejar los datos de tiempo de entrega de artículos."""

    def __init__(self, default_leadtime=15):
        """
        Inicializa el manejador de tiempos de entrega.

        Args:
            default_leadtime: Tiempo de entrega por defecto en días para artículos no encontrados
        """
        self.leadtimes = {}
        self.default_leadtime = default_leadtime

    def load_from_excel(self, file_path):
        """
        Carga datos de tiempo de entrega desde un archivo Excel.

        Args:
            file_path: Ruta al archivo Excel

        Returns:
            dict: Diccionario con los tiempos de entrega
        """
        if not os.path.exists(file_path):
            logger.error(f"El archivo {file_path} no existe.")
            return {}

        try:
            # Leer el archivo Excel
            df = pd.read_excel(file_path)

            # Detectar columnas específicas basadas en los encabezados
            codigo_col = None
            descripcion_col = None
            leadtime_col = None

            for col in df.columns:
                col_lower = col.lower()
                if "código" in col_lower or "codigo" in col_lower:
                    codigo_col = col
                elif (
                    "descripción" in col_lower
                    or "descripcion" in col_lower
                    or "artículo" in col_lower
                    or "articulo" in col_lower
                ):
                    descripcion_col = col
                elif (
                    "lead time" in col_lower
                    or "tiempo" in col_lower
                    or "dias" in col_lower
                    or "días" in col_lower
                ):
                    leadtime_col = col

            # Si no encontramos exactamente las columnas, usar posiciones
            if (
                codigo_col is None
                or descripcion_col is None
                or leadtime_col is None
            ):
                logger.warning(
                    "No se encontraron las columnas exactas, usando posiciones por defecto"
                )
                if len(df.columns) >= 4:
                    codigo_col = df.columns[0]
                    descripcion_col = df.columns[1]
                    leadtime_col = df.columns[
                        3
                    ]  # La columna D suele ser la 4ta (índice 3)
                else:
                    logger.error("Formato de archivo no reconocido")
                    return {}

            # Limpiar y cargar los datos en el diccionario
            leadtimes_count = 0
            for _, row in df.iterrows():
                # Intentar primero con el código
                if pd.notna(row[codigo_col]) and pd.notna(row[leadtime_col]):
                    articulo_id = str(row[codigo_col]).strip()
                    leadtime = float(row[leadtime_col])
                    self.leadtimes[articulo_id] = int(leadtime)
                    leadtimes_count += 1

                # También guardar por descripción para facilitar coincidencias
                if pd.notna(row[descripcion_col]) and pd.notna(
                    row[leadtime_col]
                ):
                    articulo_desc = str(row[descripcion_col]).strip()
                    leadtime = float(row[leadtime_col])
                    self.leadtimes[articulo_desc] = int(leadtime)
                    # No incrementamos contador aquí para no duplicar

            logger.info(
                f"Se cargaron {leadtimes_count} tiempos de entrega desde {file_path}"
            )
            return self.leadtimes

        except Exception as e:
            logger.error(
                f"Error al cargar tiempos de entrega desde {file_path}: {str(e)}"
            )
            return {}

    def get_leadtime(self, article):
        """
        Obtiene el tiempo de entrega para un artículo.

        Args:
            article: Nombre o código del artículo

        Returns:
            int: Tiempo de entrega en días
        """
        # Buscar coincidencia exacta
        if article in self.leadtimes:
            return self.leadtimes[article]

        # Buscar coincidencia parcial si no hay exacta
        for key, value in self.leadtimes.items():
            # Verificar si el código del artículo está contenido en la clave
            if article in key or key in article:
                return value

        # Devolver valor por defecto si no se encuentra coincidencia
        logger.warning(
            f"No se encontró tiempo de entrega para {article}, usando valor por defecto: {self.default_leadtime}"
        )
        return self.default_leadtime

    def update_leadtime(self, article, leadtime):
        """
        Actualiza el tiempo de entrega para un artículo.

        Args:
            article: Nombre o código del artículo
            leadtime: Nuevo tiempo de entrega en días

        Returns:
            bool: True si se actualizó correctamente
        """
        try:
            self.leadtimes[article] = int(leadtime)
            return True
        except (ValueError, TypeError):
            logger.error(
                f"Valor de tiempo de entrega inválido para {article}: {leadtime}"
            )
            return False

    def export_to_excel(self, file_path):
        """
        Exporta los tiempos de entrega a un archivo Excel.

        Args:
            file_path: Ruta para guardar el archivo Excel

        Returns:
            bool: True si se exportó correctamente
        """
        try:
            df = pd.DataFrame(
                list(self.leadtimes.items()),
                columns=["Artículo", "Tiempo de Entrega (días)"],
            )
            df.to_excel(file_path, index=False)
            logger.info(f"Tiempos de entrega exportados a {file_path}")
            return True
        except Exception as e:
            logger.error(
                f"Error al exportar tiempos de entrega a {file_path}: {str(e)}"
            )
            return False
