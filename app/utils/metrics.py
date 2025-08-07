import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de evaluación para pronósticos.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        dict: Diccionario con métricas
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Los arrays deben tener la misma longitud")

    # Evitar divisiones por cero y valores nulos
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)

    if not valid_mask.any():
        return {
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
        }

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Calcular métricas
    mse = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_valid, y_pred_valid)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100

    # R² (Coeficiente de determinación)
    r2 = r2_score(y_true_valid, y_pred_valid)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def format_metrics(metrics_dict):
    """
    Formatea las métricas para presentación.

    Args:
        metrics_dict: Diccionario con métricas

    Returns:
        dict: Diccionario con métricas formateadas
    """
    formatted = {}

    for key, value in metrics_dict.items():
        if key == "MAPE":
            formatted[key] = f"{value:.2f}%"
        else:
            formatted[key] = f"{value:.4f}"

    return formatted
