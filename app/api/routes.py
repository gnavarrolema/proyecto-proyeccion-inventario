# app/api/routes.py
import io
import logging
import os
from datetime import datetime

import pandas as pd
from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)
from werkzeug.utils import secure_filename

from app.config import RESULTS_DIR
from app.models.forecaster import Forecaster
from app.utils.task_manager import get_task

# Inicializar blueprint
api_bp = Blueprint("api", __name__)

# Inicializar forecaster global
forecaster = Forecaster()

logger = logging.getLogger(__name__)


def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida (CSV o Excel)."""
    allowed = current_app.config.get("ALLOWED_EXTENSIONS", {"csv"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@api_bp.route("/")
def index():
    """Renderiza la página principal."""
    return render_template("index.html")


@api_bp.route("/api/load-data", methods=["POST"])
def load_data():
    """Carga y procesa un archivo de datos (CSV o Excel)."""
    try:
        if "file" not in request.files:
            return jsonify(
                {
                    "success": False,
                    "error": "No se ha proporcionado ningún archivo",
                }
            )

        file = request.files["file"]

        if file.filename == "":
            return jsonify(
                {"success": False, "error": "Nombre de archivo vacío"}
            )

        if file and allowed_file(file.filename):
            # Guardar archivo temporalmente
            filename = secure_filename(file.filename)
            file_path = os.path.join(
                current_app.config["UPLOAD_FOLDER"], filename
            )

            # Asegurar que la carpeta exista con permisos
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Intentar guardar el archivo
                try:
                    file.save(file_path)
                    logger.info(
                        f"Archivo guardado correctamente en: {file_path}"
                    )
                except PermissionError:
                    # Si hay error de permisos, usar carpeta temporal
                    import tempfile

                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, filename)
                    file.save(temp_path)
                    file_path = temp_path
                    logger.info(
                        f"Usando directorio temporal debido a problemas de "
                        f"permisos: {temp_path}"
                    )
            except Exception:
                # Si falla la creación del directorio, usar carpeta temporal
                import tempfile

                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, filename)
                file.save(temp_path)
                file_path = temp_path
                logger.info(
                    f"Usando directorio temporal debido a problemas al crear "
                    f"directorio: {temp_path}"
                )

            # Cargar datos
            data = forecaster.load_data(file_path)

            # Extraer estadísticas básicas
            stats = {
                "total_records": len(data),
                "total_articles": len(forecaster.articles),
            }

            # Manejar las fechas de manera segura
            try:
                min_date = data["fecha_std"].min()
                max_date = data["fecha_std"].max()

                if pd.notna(min_date) and pd.notna(max_date):
                    if isinstance(min_date, pd.Timestamp) or isinstance(
                        min_date, datetime
                    ):
                        stats["date_range"] = {
                            "start": min_date.strftime("%Y-%m-%d"),
                            "end": max_date.strftime("%Y-%m-%d"),
                        }
                    else:
                        stats["date_range"] = {
                            "start": str(min_date),
                            "end": str(max_date),
                        }
                else:
                    stats["date_range"] = {
                        "start": "No disponible",
                        "end": "No disponible",
                    }
            except Exception as e:
                logger.warning(f"Error al procesar fechas: {str(e)}")
                stats["date_range"] = {
                    "start": "No disponible",
                    "end": "No disponible",
                }

            # Obtener lista de artículos
            articles = [
                {"id": i, "name": article}
                for i, article in enumerate(forecaster.articles)
            ]

            return jsonify(
                {
                    "success": True,
                    "message": "Datos cargados correctamente",
                    "stats": stats,
                    "articles": articles,
                }
            )
        else:
            return jsonify(
                {"success": False, "error": "Tipo de archivo no permitido"}
            )

    except Exception as e:
        logger.error(f"Error en carga de datos: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/get-articles", methods=["GET"])
def get_articles():
    """Obtiene la lista de artículos disponibles."""
    try:
        if forecaster.articles is None:
            return jsonify(
                {"success": False, "error": "No se han cargado datos"}
            )

        articles = [
            {"id": i, "name": article}
            for i, article in enumerate(forecaster.articles)
        ]

        return jsonify({"success": True, "articles": articles})

    except Exception as e:
        logger.error(f"Error al obtener artículos: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/article-data/<int:article_id>", methods=["GET"])
def get_article_data(article_id):
    """Obtiene los datos históricos de un artículo específico."""
    try:
        if forecaster.articles is None:
            return jsonify(
                {"success": False, "error": "No se han cargado datos"}
            )

        # Convertir ID a índice
        if article_id < 0 or article_id >= len(forecaster.articles):
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]

        # Preparar series temporales
        cantidad_series = forecaster.prepare_time_series(article, "CANTIDADES")
        produccion_series = forecaster.prepare_time_series(
            article, "PRODUCCIÓN"
        )

        # Convertir a formato para gráficos
        dates = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            for d in cantidad_series.index
        ]

        data = {
            "article": article,
            "dates": dates,
            "cantidades": cantidad_series.values.tolist(),  # Cambiado de "cantidad" a "cantidades"
            "producción": produccion_series.values.tolist(),  # Cambiado de "produccion" a "producción"
        }

        return jsonify({"success": True, "data": data})

    except Exception as e:
        logger.error(f"Error al obtener datos de artículo: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/generate-forecast", methods=["POST"])
def generate_forecast():
    """Genera pronósticos para un artículo específico."""
    try:
        # Obtener parámetros
        data = request.json
        article_id = data.get("article_id")
        target = "CANTIDADES"  # Por defecto usar CANTIDADES como objetivo
        steps = data.get("steps", 6)
        # Obtener modelos seleccionados (si se proporcionan)
        selected_models = data.get(
            "models", None
        )  # Si no se especifica, usar todos

        # Validar parámetros
        if article_id is None:
            return jsonify(
                {"success": False, "error": "No se ha especificado artículo"}
            )

        # Convertir ID a índice
        article_id = int(article_id)
        if article_id < 0 or article_id >= len(forecaster.articles):
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]

        # Entrenar modelos y generar pronósticos
        results = forecaster.train_all_models(
            article, target=target, steps=steps
        )

        if not results:
            return jsonify(
                {
                    "success": False,
                    "error": "No se pudieron generar pronósticos",
                }
            )

        # Filtrar resultados por modelos seleccionados si es necesario
        if selected_models and isinstance(selected_models, list):
            filtered_results = {
                model: results[model]
                for model in results
                if model in selected_models
            }
            # Si filtramos pero no queda ningún modelo, usar todos
            if not filtered_results:
                logger.warning(
                    f"No se encontraron modelos que coincidan con "
                    f"{selected_models}, usando todos"
                )
                filtered_results = results
        else:
            filtered_results = results

        # Obtener mejor modelo entre los seleccionados
        best_model, best_score = None, float(
            "inf"
        )  # Asumir RMSE (menor es mejor)
        for model_name, model_results in filtered_results.items():
            if (
                "metrics" in model_results
                and "RMSE" in model_results["metrics"]
            ):
                score = model_results["metrics"]["RMSE"]
                if score < best_score:
                    best_score = score
                    best_model = model_name

        # Si no se pudo determinar el mejor modelo, tomar el primero disponible
        if not best_model and filtered_results:
            best_model = list(filtered_results.keys())[0]

        # Obtener métricas del mejor modelo
        if best_model and "metrics" in filtered_results[best_model]:
            best_metrics = filtered_results[best_model]["metrics"]
        else:
            best_metrics = None

        # Guardar resultados
        result_file = forecaster.save_results(article, target)

        # Preparar respuesta
        response_data = {
            "article": article,
            "target": target,
            "best_model": best_model,
            "metrics": best_metrics,
            "forecasts": {},
        }

        # Añadir pronósticos de modelos filtrados
        for model_name, model_results in filtered_results.items():
            if "forecast" in model_results:
                forecast = model_results["forecast"]
                dates = [
                    d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                    for d in forecast.index
                ]
                values = forecast.values.tolist()

                response_data["forecasts"][model_name] = {
                    "dates": dates,
                    "values": values,
                }

                if "metrics" in model_results:
                    response_data["forecasts"][model_name][
                        "metrics"
                    ] = model_results["metrics"]

        return jsonify(
            {
                "success": True,
                "message": "Pronósticos generados correctamente",
                "data": response_data,
                "result_file": result_file,
            }
        )

    except Exception as e:
        logger.error(f"Error al generar pronósticos: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/plot-forecast/<int:article_id>/<target>", methods=["GET"])
def plot_forecast(article_id, target):
    """Genera una imagen del gráfico de pronóstico."""
    try:
        if forecaster.articles is None:
            return jsonify(
                {"success": False, "error": "No se han cargado datos"}
            )

        # Convertir ID a índice
        if article_id < 0 or article_id >= len(forecaster.articles):
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]

        # Generar gráfico
        fig = forecaster.plot_forecasts(article, target)

        if fig is None:
            return jsonify(
                {"success": False, "error": "No se pudo generar el gráfico"}
            )

        # Guardar imagen en buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)

        # Enviar imagen como respuesta
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        logger.error(f"Error al generar gráfico: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/export-forecast/<int:article_id>/<target>", methods=["GET"])
def export_forecast(article_id, target):
    """Exporta los pronósticos a un archivo CSV."""
    try:
        if forecaster.articles is None:
            return jsonify(
                {"success": False, "error": "No se han cargado datos"}
            )

        # Convertir ID a índice
        if article_id < 0 or article_id >= len(forecaster.articles):
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]

        # Verificar si tenemos resultados
        key = f"{article}_{target}"
        if key not in forecaster.results:
            return jsonify(
                {
                    "success": False,
                    "error": "No hay pronósticos para este artículo",
                }
            )

        # Obtener mejor modelo
        best_model, forecast, _ = forecaster.get_best_model(article, target)

        if forecast is None:
            return jsonify(
                {"success": False, "error": "No hay pronóstico disponible"}
            )

        # Exportar a CSV
        filename = f"pronostico_{article_id}_{target}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        file_path = os.path.join(RESULTS_DIR, filename)

        # Crear DataFrame
        import pandas as pd

        df = pd.DataFrame(
            {
                "fecha": [
                    d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                    for d in forecast.index
                ],
                "articulo": article,
                "modelo": best_model,
                "valor": forecast.values,
            }
        )

        # Guardar a CSV
        df.to_csv(file_path, index=False)

        # Enviar archivo
        return send_file(file_path, as_attachment=True, download_name=filename)

    except Exception as e:
        logger.error(f"Error al exportar pronóstico: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/optimize-xgboost", methods=["POST"])
def optimize_xgboost():
    """Inicia la optimización de XGBoost en segundo plano."""
    try:
        # Log inicial para verificar que la ruta se está llamando
        logger.info("Solicitud de optimización XGBoost recibida")

        # Obtener parámetros
        data = request.json
        article_id = data.get("article_id")
        target = data.get("target", "CANTIDADES")
        n_trials = data.get("n_trials", 50)

        logger.info(
            f"Parámetros de optimización: article_id={article_id}, "
            f"target={target}, n_trials={n_trials}"
        )

        # Validar parámetros
        if article_id is None:
            logger.error("No se ha especificado artículo")
            return jsonify(
                {"success": False, "error": "No se ha especificado artículo"}
            )

        # Convertir ID a índice
        try:
            article_id = int(article_id)
        except (ValueError, TypeError):
            logger.error(f"ID de artículo inválido: {article_id}")
            return jsonify(
                {"success": False, "error": "ID de artículo inválido"}
            )

        # Verificar que hay datos cargados
        if forecaster.articles is None or len(forecaster.articles) == 0:
            logger.error("No hay datos cargados en el forecaster")
            return jsonify(
                {
                    "success": False,
                    "error": "No hay datos cargados. Por favor, cargue un "
                    "archivo CSV primero.",
                }
            )

        if article_id < 0 or article_id >= len(forecaster.articles):
            logger.error(
                f"Artículo no encontrado: índice {article_id} está fuera de "
                f"rango (0-{len(forecaster.articles)-1})"
            )
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]
        logger.info(f"Artículo para optimización: {article}")

        # Verificar que hay suficientes datos para optimizar
        series = forecaster.prepare_time_series(article, target)
        if len(series) < 12:
            logger.error(
                f"Serie temporal insuficiente para optimización: {len(series)} "
                f"puntos, se requieren al menos 12"
            )
            return jsonify(
                {
                    "success": False,
                    "error": f"Serie temporal insuficiente para "
                    f"optimización: {len(series)} puntos, se requieren al "
                    f"menos 12",
                }
            )

        # Definir función para optimización en segundo plano
        def optimize_task(forecaster, article, target, n_trials, task_id):
            logger.info(
                f"Tarea de optimización iniciada para {article} "
                f"(task_id: {task_id})"
            )

            def progress_callback(progress, current_trial, total_trials):
                # Obtener tarea del almacén y actualizar progreso
                from app.utils.task_manager import TaskStatus, task_store

                if task_id in task_store:
                    try:
                        logger.info(
                            f"Actualización de progreso: {progress}% ({current_trial}/{total_trials})"
                        )
                        task_store[task_id].update(
                            progress=progress,
                            result={
                                "current_trial": current_trial,
                                "total_trials": total_trials,
                            },
                        )
                    except Exception as e:
                        logger.error(f"Error al actualizar progreso: {str(e)}")
                else:
                    logger.error(
                        f"Task ID {task_id} no encontrado en task_store"
                    )

            # Ejecutar optimización
            try:
                logger.info(
                    f"Iniciando optimize_xgboost_with_progress para {article}"
                )
                best_params = forecaster.optimize_xgboost_with_progress(
                    article,
                    target=target,
                    n_trials=n_trials,
                    progress_callback=progress_callback,
                )

                if best_params:
                    logger.info(
                        f"Optimización completada con éxito para {article}. "
                        f"Mejores parámetros: {best_params}"
                    )
                else:
                    logger.warning(
                        f"La optimización no encontró parámetros óptimos para "
                        f"{article}"
                    )

                # Generar pronóstico con el modelo optimizado
                forecast = None
                try:
                    if best_params:
                        logger.info(
                            f"Generando pronóstico con parámetros optimizados para {article}"
                        )
                        forecast_model = (
                            forecaster.train_xgboost_with_optimal_params(
                                article, target
                            )
                        )

                        if (
                            isinstance(forecast_model, pd.Series)
                            and not forecast_model.empty
                        ):
                            dates = [
                                d.strftime("%Y-%m-%d")
                                if hasattr(d, "strftime")
                                else str(d)
                                for d in forecast_model.index
                            ]
                            values = forecast_model.values.tolist()
                            forecast = {"dates": dates, "values": values}
                            logger.info(
                                f"Pronóstico generado con éxito para {article}"
                            )
                except Exception as forecast_error:
                    logger.error(
                        f"Error al generar pronóstico con parámetros optimizados: {str(forecast_error)}"
                    )

                return {"best_params": best_params, "forecast": forecast}

            except Exception as e:
                logger.error(
                    f"Error durante la optimización: {str(e)}", exc_info=True
                )
                return {"error": str(e), "best_params": None}

        # Iniciar tarea en segundo plano con un ID único basado en timestamp
        import time

        task_id = f"optimize_xgboost_{article_id}_{target}_{int(time.time())}"

        logger.info(f"Iniciando tarea en segundo plano con ID: {task_id}")

        # Iniciar la tarea en segundo plano
        from app.utils.task_manager import run_task_in_background

        task_id = run_task_in_background(
            optimize_task,
            task_id,
            forecaster,
            article,
            target,
            n_trials,
            task_id,
        )

        logger.info(f"Tarea iniciada con ID: {task_id}")

        return jsonify(
            {
                "success": True,
                "message": "Optimización iniciada en segundo plano",
                "task_id": task_id,
            }
        )

    except Exception as e:
        import traceback

        logger.error(f"Error al iniciar optimización de XGBoost: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/optimize-xgboost-status/<task_id>", methods=["GET"])
def optimize_xgboost_status(task_id):
    """Obtiene el estado actual de una tarea de optimización."""
    try:
        task_info = get_task(task_id)

        if not task_info:
            return jsonify({"success": False, "error": "Tarea no encontrada"})

        return jsonify({"success": True, "task": task_info})

    except Exception as e:
        logger.error(f"Error al obtener estado de optimización: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/debug-forecast/<int:article_id>/<target>", methods=["GET"])
def debug_forecast(article_id, target):
    """Endpoint de depuración para verificar los datos de pronóstico."""
    try:
        if forecaster.articles is None:
            return jsonify(
                {"success": False, "error": "No se han cargado datos"}
            )

        # Convertir ID a índice
        if article_id < 0 or article_id >= len(forecaster.articles):
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]

        # Obtener datos históricos
        series = forecaster.prepare_time_series(article, target)

        # Convertir a formato para inspección
        dates_info = {
            "index_type": str(type(series.index)),
            "first_date_type": str(type(series.index[0]))
            if len(series.index) > 0
            else "N/A",
            "first_date_value": str(series.index[0])
            if len(series.index) > 0
            else "N/A",
            "date_sample": [str(d) for d in series.index[:5].tolist()]
            if len(series.index) > 5
            else [str(d) for d in series.index.tolist()],
        }

        # Información del pronóstico
        forecast_info = {}

        # Usar try/except aquí para evitar que un error durante el entrenamiento interrumpa la depuración
        try:
            results = forecaster.train_all_models(
                article, target=target, steps=3, debug=True
            )

            for model_name, model_results in results.items():
                if "forecast" in model_results:
                    forecast = model_results["forecast"]
                    forecast_info[model_name] = {
                        "index_type": str(type(forecast.index)),
                        "first_date_type": str(type(forecast.index[0]))
                        if len(forecast.index) > 0
                        else "N/A",
                        "first_date_value": str(forecast.index[0])
                        if len(forecast.index) > 0
                        else "N/A",
                        "date_sample": [
                            str(d) for d in forecast.index.tolist()
                        ],
                        "value_sample": forecast.values.tolist(),
                    }
        except Exception as e:
            forecast_info = {"error": str(e)}

        return jsonify(
            {
                "success": True,
                "article": article,
                "target": target,
                "historical_data": {
                    "dates_info": dates_info,
                    "values_sample": series.values[:5].tolist()
                    if len(series) > 5
                    else series.values.tolist(),
                },
                "forecast_info": forecast_info,
            }
        )

    except Exception as e:
        logger.error(f"Error en depuración: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@api_bp.route("/api/debug-xgboost-optimize/<int:article_id>", methods=["GET"])
def debug_xgboost_optimize(article_id):
    """Endpoint de diagnóstico para la optimización XGBoost."""
    try:
        if article_id < 0 or article_id >= len(forecaster.articles):
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]
        target = "CANTIDADES"

        # Verificar si hay suficientes datos
        series = forecaster.prepare_time_series(article, target)

        # Información del módulo Optuna
        import optuna

        optuna_info = {"version": optuna.__version__, "available": True}

        # Verificar disponibilidad de XGBoost
        xgboost_info = {"available": False}
        try:
            import xgboost as xgb

            xgboost_info = {"version": xgb.__version__, "available": True}
        except Exception:
            pass

        # Guardar diagnóstico
        diagnostics = {
            "article": article,
            "series_length": len(series),
            "series_stats": {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            },
            "optuna_info": optuna_info,
            "xgboost_info": xgboost_info,
            "forecaster_models_count": len(forecaster.models),
            "has_hyperparameter_tuner": hasattr(
                forecaster, "optimize_xgboost_with_progress"
            ),
        }

        return jsonify({"success": True, "diagnostics": diagnostics})
    except Exception as e:
        import traceback

        return jsonify(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


@api_bp.route("/api/optimize-model", methods=["POST"])
def optimize_model():
    """Inicia la optimización de cualquier modelo en segundo plano."""
    try:
        # Obtener parámetros
        data = request.json
        article_id = data.get("article_id")
        model_type = data.get(
            "model_type"
        )  # 'SARIMA', 'LSTM', 'GRU', 'XGBOOST'
        target = data.get("target", "CANTIDADES")
        n_trials = data.get("n_trials", 50)

        logger.info(
            f"Solicitud de optimización de {model_type} recibida para artículo ID {article_id}"
        )

        # Validar parámetros
        if article_id is None:
            return jsonify(
                {"success": False, "error": "No se ha especificado artículo"}
            )

        if model_type not in ["SARIMA", "LSTM", "GRU", "XGBOOST"]:
            return jsonify(
                {
                    "success": False,
                    "error": f"Tipo de modelo no válido: {model_type}. Debe ser SARIMA, LSTM, GRU o XGBOOST",
                }
            )

        # Convertir ID a índice
        try:
            article_id = int(article_id)
        except (ValueError, TypeError):
            return jsonify(
                {"success": False, "error": "ID de artículo inválido"}
            )

        # Verificar que hay datos cargados
        if forecaster.articles is None or len(forecaster.articles) == 0:
            return jsonify({"success": False, "error": "No hay datos cargados"})

        if article_id < 0 or article_id >= len(forecaster.articles):
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]

        # Función a ejecutar en segundo plano
        def optimize_task(
            forecaster, article, model_type, target, n_trials, task_id
        ):
            logger.info(
                f"Tarea de optimización de {model_type} iniciada para {article} (task_id: {task_id})"
            )

            def progress_callback(progress, current_trial, total_trials):
                # Obtener tarea del almacén y actualizar progreso
                from app.utils.task_manager import TaskStatus, task_store

                if task_id in task_store:
                    try:
                        logger.info(
                            f"Actualización de progreso: {progress}% ({current_trial}/{total_trials})"
                        )
                        task_store[task_id].update(
                            progress=progress,
                            result={
                                "current_trial": current_trial,
                                "total_trials": total_trials,
                            },
                        )
                    except Exception as e:
                        logger.error(f"Error al actualizar progreso: {str(e)}")
                else:
                    logger.error(
                        f"Task ID {task_id} no encontrado en task_store"
                    )

            # Ejecutar optimización
            try:
                best_params = forecaster.optimize_model_hyperparameters(
                    article,
                    model_type,
                    target=target,
                    n_trials=n_trials,
                    progress_callback=progress_callback,
                )

                if best_params:
                    logger.info(
                        f"Optimización completada con éxito para {article} ({model_type})"
                    )

                    # Generar pronóstico con el modelo optimizado
                    forecast = None
                    try:
                        if best_params:
                            logger.info(
                                f"Generando pronóstico con parámetros optimizados para {article}"
                            )

                            # Aquí deberíamos implementar la generación de pronósticos con el modelo optimizado
                            # Esto dependerá de cada tipo de modelo

                    except Exception as forecast_error:
                        logger.error(
                            f"Error al generar pronóstico con parámetros "
                            f"optimizados: {str(forecast_error)}"
                        )

                return {"best_params": best_params, "model_type": model_type}

            except Exception as e:
                logger.error(
                    f"Error durante la optimización: {str(e)}", exc_info=True
                )
                return {"error": str(e), "best_params": None}

        # Iniciar tarea en segundo plano con un ID único
        import time

        task_id = (
            f"optimize_{model_type}_{article_id}_{target}_{int(time.time())}"
        )

        # Iniciar la tarea en segundo plano
        from app.utils.task_manager import run_task_in_background

        task_id = run_task_in_background(
            optimize_task,
            task_id,
            forecaster,
            article,
            model_type,
            target,
            n_trials,
            task_id,
        )

        logger.info(f"Tarea iniciada con ID: {task_id}")

        return jsonify(
            {
                "success": True,
                "message": f"Optimización de {model_type} iniciada en segundo plano",
                "task_id": task_id,
            }
        )

    except Exception as e:
        logger.error(f"Error al iniciar optimización de {model_type}: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


# Ruta para cargar datos de tiempos de entrega
@api_bp.route("/api/load-leadtimes", methods=["POST"])
def load_leadtimes():
    """Carga datos de tiempos de entrega desde un archivo Excel."""
    try:
        # Logging para depuración
        logger.info("Endpoint /api/load-leadtimes llamado")

        if "file" not in request.files:
            logger.error("No se encontró archivo en la solicitud")
            return jsonify(
                {
                    "success": False,
                    "error": "No se ha proporcionado ningún archivo",
                }
            )

        file = request.files["file"]

        if file.filename == "":
            return jsonify(
                {"success": False, "error": "Nombre de archivo vacío"}
            )

        # Validar extensión
        if not file.filename.lower().endswith((".xls", ".xlsx")):
            return jsonify(
                {
                    "success": False,
                    "error": "Tipo de archivo no permitido. Use Excel (.xls, .xlsx)",
                }
            )

        # Guardar temporalmente usando tempfile
        import tempfile

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))

        try:
            file.save(temp_path)
            logger.info(f"Archivo guardado temporalmente en: {temp_path}")
        except Exception as e:
            logger.error(f"Error al guardar archivo: {str(e)}")
            return jsonify(
                {
                    "success": False,
                    "error": f"Error al guardar archivo: {str(e)}",
                }
            )

        # Cargar tiempos de entrega
        try:
            leadtimes = forecaster.leadtime_handler.load_from_excel(temp_path)
            count = len(leadtimes)

            # Limpiar archivo temporal
            try:
                os.remove(temp_path)
            except:
                pass

            return jsonify(
                {
                    "success": True,
                    "message": f"Tiempos de entrega cargados correctamente: {count} artículos",
                    "count": count,
                }
            )
        except Exception as e:
            logger.error(f"Error al procesar archivo: {str(e)}")
            return jsonify({"success": False, "error": str(e)})

    except Exception as e:
        logger.error(f"Error general en load_leadtimes: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


def _convert_numpy_types_for_json(obj):
    """Convierte tipos de NumPy a tipos nativos de Python para serialización JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _convert_numpy_types_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types_for_json(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_numpy_types_for_json(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Convierte numpy.bool_ a bool nativo de Python
    else:
        return obj


# Ruta para calcular stock de seguridad para un artículo
@api_bp.route("/api/safety-stock/<int:article_id>", methods=["GET"])
def calculate_safety_stock(article_id):
    """Calcula el stock de seguridad para un artículo específico."""
    try:
        if forecaster.articles is None:
            return jsonify(
                {"success": False, "error": "No se han cargado datos"}
            )

        # Convertir ID a índice
        if article_id < 0 or article_id >= len(forecaster.articles):
            return jsonify(
                {"success": False, "error": "Artículo no encontrado"}
            )

        article = forecaster.articles[article_id]

        # Obtener parámetros de la solicitud
        target = request.args.get("target", "CANTIDADES")
        method = request.args.get("method", "basic")
        service_level = request.args.get("service_level")
        if service_level is not None:
            service_level = float(service_level)
        else:
            service_level = 0.95

        # Obtener horizonte de pronóstico y si usar pronósticos
        forecast_horizon = request.args.get("forecast_horizon")
        if forecast_horizon is not None:
            forecast_horizon = int(forecast_horizon)
        else:
            forecast_horizon = 6  # 6 meses por defecto

        use_forecasts = (
            request.args.get("use_forecasts", "true").lower() == "true"
        )

        # Registrar los parámetros para depuración
        logger.info(
            f"Calculando stock de seguridad para {article} con parámetros: target={target}, method={method}, service_level={service_level}, forecast_horizon={forecast_horizon}, use_forecasts={use_forecasts}"
        )

        # Calcular stock de seguridad
        result = forecaster.calculate_safety_stock(
            article,
            target,
            method,
            service_level,
            None,
            forecast_horizon=forecast_horizon,
            use_forecasts=use_forecasts,
        )

        # Convertir tipos de NumPy a tipos nativos para serialización JSON
        result = _convert_numpy_types_for_json(result)

        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(
            f"Error al calcular stock de seguridad: {str(e)}", exc_info=True
        )
        return jsonify({"success": False, "error": str(e)})


# Ruta para calcular stock de seguridad para todos los artículos
@api_bp.route("/api/safety-stock", methods=["GET"])
def calculate_all_safety_stocks():
    """Calcula el stock de seguridad para todos los artículos."""
    try:
        if forecaster.articles is None:
            return jsonify(
                {"success": False, "error": "No se han cargado datos"}
            )

        # Obtener parámetros de la solicitud
        target = request.args.get("target", "CANTIDADES")
        method = request.args.get("method", "basic")
        service_level = request.args.get("service_level")
        if service_level is not None:
            service_level = float(service_level)
        else:
            service_level = 0.95

        # Obtener horizonte de pronóstico y si usar pronósticos
        forecast_horizon = request.args.get("forecast_horizon")
        if forecast_horizon is not None:
            forecast_horizon = int(forecast_horizon)
        else:
            forecast_horizon = 6  # 6 meses por defecto

        use_forecasts = (
            request.args.get("use_forecasts", "true").lower() == "true"
        )

        # Calcular stock de seguridad para todos los artículos
        results = forecaster.calculate_all_safety_stocks(
            target,
            method,
            service_level,
            forecast_horizon=forecast_horizon,
            use_forecasts=use_forecasts,
        )

        # Convertir a formato para respuesta JSON
        formatted_results = {}
        for article, result in results.items():
            formatted_result = {
                "safety_stock": result.get("safety_stock", 0),
                "leadtime_days": result.get("leadtime_days", 0),
                "service_level": result.get("service_level", 0),
                "method": result.get("method", ""),
                "data_points": result.get("data_points", 0),
                "avg_demand": float(result.get("avg_demand", 0)),
                "std_dev": float(result.get("std_dev", 0)),
            }

            # Añadir stock de seguridad por mes si está disponible
            if "safety_stocks_by_month" in result:
                formatted_result["safety_stocks_by_month"] = result[
                    "safety_stocks_by_month"
                ]

            # Añadir información del mejor modelo si está disponible
            if "best_model" in result:
                formatted_result["best_model"] = result["best_model"]

            formatted_results[article] = formatted_result

        return jsonify({"success": True, "data": formatted_results})
    except Exception as e:
        logger.error(
            f"Error al calcular stock de seguridad para todos los artículos: {str(e)}"
        )
        return jsonify({"success": False, "error": str(e)})
