import time
import json
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import os
import importlib
import inspect
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from .metrics import MetricsEngine
from .ranker import WeightedRanker


class ModelRunner:
    def __init__(self):
        # ── Keys MUST match the names used in master_rules.yaml and engine.py ──
        self.model_catalog = {
            "LogReg":        LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest":  RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM":           SVC(probability=True, random_state=42),
            "KNN":           KNeighborsClassifier(),
            # MultinomialNB works on sparse TF-IDF matrices; GaussianNB does not
            "NaiveBayes":    MultinomialNB(),
            "DecisionTree":  DecisionTreeClassifier(random_state=42),
        }

        if xgb:
            try:
                self.model_catalog["XGBoost"] = xgb.XGBClassifier(
                    eval_metric='logloss',
                    tree_method='hist',
                    device='cuda',
                    random_state=42
                )
            except Exception:
                # Fall back to CPU if CUDA isn't available
                self.model_catalog["XGBoost"] = xgb.XGBClassifier(
                    eval_metric='logloss',
                    tree_method='hist',
                    random_state=42
                )

        self._load_custom_models()

    def _load_custom_models(self):
        custom_dir = Path(__file__).resolve().parent.parent / "custom_models"
        if not custom_dir.exists():
            return
        for filename in os.listdir(custom_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"custom_models.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, "fit") and hasattr(obj, "predict"):
                            self.model_catalog[f"Custom:{name}"] = obj()
                            print(f"[BYOM] Armed custom model: {name}")
                except Exception as e:
                    print(f"[BYOM] Error loading {filename}: {e}")

    def _train_single_model(self, name, model, X_train, X_test, y_train, y_test, classes):
        start_time = time.perf_counter()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = round(float(np.mean(cv_scores)), 4)
        cv_std  = round(float(np.std(cv_scores)), 4)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        train_time = time.perf_counter() - start_time

        metrics    = MetricsEngine.calculate_classification(y_test, y_pred, y_prob, classes)
        efficiency = MetricsEngine.calculate_efficiency(model, X_test, train_time)

        return {
            "name": name,
            "metrics": metrics,
            "efficiency": efficiency,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

    def train_and_evaluate(self, active_models: list, X_train, X_test, y_train, y_test):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test  = le.transform(y_test)
        classes = [str(c) for c in le.classes_]

        # Guard: nothing survived Layer 3
        if not active_models:
            raise ValueError(
                "Layer 3 excluded ALL models for this dataset. "
                "Consider relaxing the filter rules or trying a different task type."
            )

        models_to_run = [
            (name, self.model_catalog[name])
            for name in active_models
            if name in self.model_catalog
        ]

        # Guard: names didn't match catalog (shouldn't happen now, but safety net)
        if not models_to_run:
            available = list(self.model_catalog.keys())
            raise ValueError(
                f"No catalog match for active models {active_models}. "
                f"Catalog has: {available}"
            )

        print(f"[LAYER 5] Training {len(models_to_run)} models sequentially...")

        parallel_results = []
        for name, model in models_to_run:
            print(f"[pipeline] Training {name}...")
            try:
                res = self._train_single_model(
                    name, model, X_train, X_test, y_train, y_test, classes
                )
                parallel_results.append(res)
            except Exception as e:
                print(f"[pipeline] {name} failed: {e} — skipping")

        if not parallel_results:
            raise ValueError("All models failed during training. Check the dataset and preprocessing.")

        final_models = {}
        for res in parallel_results:
            name = res.pop("name")
            final_models[name] = res

        ranker = WeightedRanker(task_type="classification")
        ranked_models_list = ranker.score_models(final_models)
        final_report = ranker.generate_report(ranked_models_list, classes)

        return final_report

    def save_results(self, results_dict, filepath):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)