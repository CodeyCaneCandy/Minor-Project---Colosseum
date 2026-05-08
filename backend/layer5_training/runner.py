import time
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import os
import importlib
import inspect
from pathlib import Path

# Import the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# --- IMPORT OUR NEW MODULAR ENGINES ---
from .metrics import MetricsEngine
from .ranker import WeightedRanker

class ModelRunner:
    def __init__(self):
        # The Master Roster of Models
        self.model_catalog = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }
        if xgb:
            self.model_catalog["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
        # --- NEW: SCAN FOR CUSTOM PLUGINS ---
        self._load_custom_models()

    def _load_custom_models(self):
        """Scans the custom_models directory and dynamically injects them into the Gauntlet."""
        # Find the backend/custom_models directory
        custom_dir = Path(__file__).resolve().parent.parent / "custom_models"
        
        if not custom_dir.exists():
            return
            
        for filename in os.listdir(custom_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"custom_models.{filename[:-3]}"
                try:
                    # Dynamically import the Python file
                    module = importlib.import_module(module_name)
                    
                    # Scan the file for any Classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Duck-typing check: Does the class have a .fit() and .predict() method?
                        if hasattr(obj, "fit") and hasattr(obj, "predict"):
                            # Instantiate it and add it to the roster!
                            self.model_catalog[f"Custom: {name}"] = obj()
                            print(f"[BYOM] Successfully armed custom model: {name}")
                except Exception as e:
                    print(f"[BYOM] Error loading plugin {filename}: {e}")
        
    def _train_single_model(self, name, model, X_train, X_test, y_train, y_test, classes):
        """Worker function that runs on an isolated CPU/GPU core."""
        start_time = time.perf_counter()
        
        # 1. Cross-Validation (The Practice Exam)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = round(float(np.mean(cv_scores)), 4)
        cv_std = round(float(np.std(cv_scores)), 4)
        
        # 2. Main Fit (The Real Training)
        model.fit(X_train, y_train)
        
        # 3. Inference (The Final Exam)
        y_pred = model.predict(X_test)
        
        # Attempt to get probabilities for ROC-AUC
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
        train_time = time.perf_counter() - start_time
        
        # 4. Offload Math to MetricsEngine
        metrics = MetricsEngine.calculate_classification(y_test, y_pred, y_prob, classes)
        efficiency = MetricsEngine.calculate_efficiency(model, X_test, train_time)
        
        return {
            "name": name,
            "metrics": metrics,
            "efficiency": efficiency,
            "cv_mean": cv_mean,
            "cv_std": cv_std
        }

    def train_and_evaluate(self, active_models: list, X_train, X_test, y_train, y_test):
        """Orchestrates the parallel training and final ranking."""
        
        # --- THE UNIVERSAL LABEL ENCODER ---
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        classes = [str(c) for c in le.classes_]
        
        # Filter catalog to only include models that survived Layer 3
        models_to_run = [
            (name, self.model_catalog[name]) 
            for name in active_models if name in self.model_catalog
        ]
        
        print(f"[LAYER 5] Dispatching {len(models_to_run)} models to parallel worker pool...")
        
        # Run all models in parallel!
        parallel_results = Parallel(n_jobs=-1)(
            delayed(self._train_single_model)(
                name, model, X_train, X_test, y_train, y_test, classes
            ) for name, model in models_to_run
        )
        
        # Reformat the parallel results into a dictionary
        final_models = {}
        for res in parallel_results:
            name = res.pop("name")
            final_models[name] = res
            
        # --- OFFLOAD BUSINESS LOGIC TO RANKER ---
        ranker = WeightedRanker(task_type="classification")
        ranked_models_list = ranker.score_models(final_models)
        final_report = ranker.generate_report(ranked_models_list, classes)
        
        return final_report

    def save_results(self, results_dict, filepath):
        """Saves the final JSON payload to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)