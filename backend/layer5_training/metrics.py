import time
import os
import tempfile
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)

class MetricsEngine:
    @staticmethod
    def calculate_classification(y_true, y_pred, y_prob=None, classes=None):
        """Calculates all standard classification metrics."""
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted')
        
        # Safe ROC-AUC calculation (handles binary vs multiclass gracefully)
        roc = 0.0
        if y_prob is not None:
            try:
                if len(classes) == 2:
                    roc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    roc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception:
                roc = 0.0 # Fallback if probability distributions are missing classes

        return {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "roc_auc": round(roc, 4),
            "cm": confusion_matrix(y_true, y_pred).tolist()
        }

    @staticmethod
    def calculate_regression(y_true, y_pred):
        """Calculates standard regression metrics."""
        return {
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
            "mse": round(mean_squared_error(y_true, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            "r2": round(r2_score(y_true, y_pred), 4),
            "mape": round(mean_absolute_percentage_error(y_true, y_pred), 4)
        }

    @staticmethod
    def calculate_efficiency(model, X_test, train_time):
        """Calculates prediction latency and serialized footprint."""
        # 1. Latency: Predict exactly 1 sample to see how fast inference is
        single_sample = X_test[0:1]
        start = time.perf_counter()
        model.predict(single_sample)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # 2. Disk Size: Dump to a temp file to check RAM/Disk weight
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            joblib.dump(model, tmp.name)
            size_mb = round(os.path.getsize(tmp.name) / (1024 * 1024), 3)
        os.remove(tmp.name)

        return {
            "train_time_sec": round(train_time, 3),
            "prediction_latency_ms": latency_ms,
            "size_mb": size_mb
        }