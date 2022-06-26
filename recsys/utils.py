import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(y_true, y_pred, rmse=True, mae=True):
    """Считает значения метрик качества.

    Args:
        y_true: Истинные значения (таргеты).
        y_pred: Предсказанные значения.
        rmse: Включение расчета RMSE.
        mae: Включение расчета MAE.
    """
    metrics = dict()
    if rmse:
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    if mae:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
    return metrics
