from typing import Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

DIMENSIONS = ["clarity", "relevance", "informativeness", "structure"]


def summarize_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for i, dim in enumerate(DIMENSIONS):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        stat = spearmanr(y_true, y_pred).statistic
        if stat is None or np.isnan(stat):
            stat = 0.0
        results[dim] = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'spearman': float(stat),
        }
    results['macro_avg'] = {
        'mae': float(np.mean([results[d]['mae'] for d in DIMENSIONS])),
        'mse': float(np.mean([results[d]['mse'] for d in DIMENSIONS])),
        'spearman': float(np.mean([results[d]['spearman'] for d in DIMENSIONS])),
    }
    return results
