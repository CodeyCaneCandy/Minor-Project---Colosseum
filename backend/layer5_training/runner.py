import json
import time
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def _train_single_model(model_name, model, X_train, X_test, y_train, y_test, classes):
    start_time = time.time()

    # FIX 1: Sparse matrix handling expanded.
    # GaussianNB requires dense. KNN and SVM are very slow on sparse — convert them too.
    # XGBoost, LogReg, RandomForest, DecisionTree handle sparse natively — leave them alone.
    NEEDS_DENSE = {"NaiveBayes", "KNN", "SVM"}
    if sp.issparse(X_train) and model_name in NEEDS_DENSE:
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # FIX 2: cross_val_score clones the model internally, so it's safe to run before fit.
    # But we need to guard against models that can't handle the data shape at CV time too.
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
        cv_mean = round(float(np.mean(cv_scores)), 4)
        cv_std = round(float(np.std(cv_scores)), 4)
    except Exception as cv_err:
        print(f"[CV WARNING] {model_name} cross-validation failed: {cv_err}. Skipping CV.")
        cv_mean = 0.0
        cv_std = 0.0

    # Main training on the full training split
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 3)

    y_pred = model.predict(X_test)

    # FIX 3: bare except replaced with explicit Exception catch so KeyboardInterrupt still propagates
    try:
        if len(classes) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_prob)
        else:
            y_prob = model.predict_proba(X_test)
            roc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except Exception:
        roc = 0.0

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted')

    composite = round((f1 * 0.7) + (acc * 0.3), 4)

    return {
        "name": model_name,
        "metrics": {
            "accuracy":        round(float(acc),  4),
            "f1":              round(float(f1),   4),
            "precision":       round(float(prec), 4),
            "recall":          round(float(rec),  4),
            "roc_auc":         round(float(roc),  4),
            "composite_score": composite,
            "cv_mean":         cv_mean,
            "cv_std":          cv_std,
        },
        "time": train_time,
        "cm":   confusion_matrix(y_test, y_pred).tolist()
    }


class ModelRunner:
    def __init__(self):
        # FIX 4: use_label_encoder was removed in XGBoost 1.6 — drop it to avoid warnings
        xgb_params = {'eval_metric': 'logloss', 'tree_method': 'hist'}

        try:
            import xgboost as xgb
            test_model = xgb.XGBClassifier(tree_method='hist', device='cuda')
            test_model.fit(np.zeros((2, 1)), np.array([0, 1]))  # FIX 5: need ≥2 samples to fit
            xgb_params['device'] = 'cuda'
            print("\n[SYSTEM 🟢] NVIDIA GPU Detected! XGBoost subscribed to CUDA.")
        except Exception:
            print("\n[SYSTEM 🟡] No GPU detected or configured. Defaulting to CPU.")

        self.model_catalog = {
            "LogReg":       LogisticRegression(max_iter=1000),
            "SVM":          SVC(probability=True),
            "NaiveBayes":   GaussianNB(),
            "KNN":          KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(n_jobs=-1),  # FIX 6: RF can parallelise internally
            "XGBoost":      XGBClassifier(**xgb_params)
        }

    def train_and_evaluate(self, active_models: list, X_train, X_test, y_train, y_test) -> dict:
        
        # --- THE UNIVERSAL LABEL ENCODER ---
        # XGBoost and Neural Networks strictly require classes to start at 0.
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        classes = [str(c) for c in le.classes_]

        models_to_run = [
            (name, self.model_catalog[name])
            for name in active_models if name in self.model_catalog
        ]
        print(f"[LAYER 4] Dispatching {len(models_to_run)} models to parallel worker pool...")
        # ... rest of the code remains exactly the same
        
        if not models_to_run:
            # FIX 7: guard against Layer 3 excluding everything
            raise ValueError("Layer 3 excluded all models. No models left to train.")

        print(f"[LAYER 5] Dispatching {len(models_to_run)} models to parallel worker pool...")
        parallel_results = Parallel(n_jobs=-1)(
            delayed(_train_single_model)(name, model, X_train, X_test, y_train, y_test, classes)
            for name, model in models_to_run
        )

        final_models = {}
        final_times  = {}
        best_score   = -1   # FIX 8: init to -1 so a 0.0 score can still win
        winner_name  = ""
        best_cm      = None

        for res in parallel_results:
            final_models[res["name"]] = res["metrics"]
            final_times[res["name"]]  = res["time"]

            if res["metrics"]["composite_score"] > best_score:
                best_score  = res["metrics"]["composite_score"]
                winner_name = res["name"]
                best_cm     = res["cm"]

        # FIX 9: confidence band has three levels, not two
        if best_score > 0.85:
            conf_level = "high"
        elif best_score > 0.70:
            conf_level = "medium"
        else:
            conf_level = "low"

        final_report = {
            "models":           final_models,
            "model_times":      final_times,
            "winner":           {"name": winner_name, "score": best_score},
            "confidence":       {"level": conf_level, "value": int(best_score * 100)},
            "explanation":      (
                f"{winner_name} outperformed the field with a composite score of "
                f"{best_score:.4f}, balancing F1 (weighted 70%) and accuracy (30%)."
            ),
            "confusion_matrix": best_cm,
            "label_classes":    classes,
        }

        return final_report

    def save_results(self, report: dict, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
