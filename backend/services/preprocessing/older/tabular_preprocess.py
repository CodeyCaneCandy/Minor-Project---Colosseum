import csv
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DynamicPreprocessor:
    """
    Tabular preprocessing pipeline for Layer 2.

    Correct order of operations
    ───────────────────────────
    load → split → fit on train → transform both
                   ↑
                   data leakage fix: scaler never sees test rows during fitting
    """

    def __init__(self, file_path: str, target_col: str):
        self.file_path      = file_path
        self.target_col     = target_col
        self.target_encoder = LabelEncoder()
        self.pipeline       = None          # set after fit

    # ── LOAD ─────────────────────────────────────────────────────────────────

    def _detect_delimiter(self) -> str:
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return ","

    def load_data(self) -> pd.DataFrame:
        sep = self._detect_delimiter()
        print(f"[load]  Detected delimiter: '{sep}'")
        df  = pd.read_csv(self.file_path, sep=sep)
        df.columns = df.columns.str.strip()

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found.\n"
                f"Available: {df.columns.tolist()}"
            )
        print(f"[load]  Shape: {df.shape}")
        return df

    # ── PIPELINE BUILD ────────────────────────────────────────────────────────

    def _build_pipeline(self, X: pd.DataFrame):
        """
        Inspect column dtypes and build a ColumnTransformer.
        Returns (pipeline, num_cols, cat_cols).
        """
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        print(f"[pipeline]  {len(num_cols)} numerical, {len(cat_cols)} categorical features")

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer",  SimpleImputer(strategy="most_frequent")),
            # handle_unknown → encodes unseen test categories as -1 instead of crashing
            ("encoder",  OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )),
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", numeric_transformer, num_cols))
        if cat_cols:
            transformers.append(("cat", categorical_transformer, cat_cols))

        pipeline = ColumnTransformer(transformers=transformers)
        return pipeline, num_cols, cat_cols

    # ── MAIN ENTRY ────────────────────────────────────────────────────────────

    def process_and_split(
        self,
        test_size:    float = 0.2,
        random_state: int   = 42,
        problem_type: str   = "classification",
    ):
        """
        Full pipeline: load → encode target → split → fit on train
                       → transform train and test separately.

        Returns
        -------
        X_train, X_test, y_train, y_test   (all as numpy arrays or DataFrames)
        pipeline                            (fitted sklearn pipeline, saved for later)
        feature_names                       (ordered list matching X columns)
        label_classes                       (list of original class names if encoded, else None)
        """
        df = self.load_data()

        # ── 1. Separate X and y ───────────────────────────────────────────────
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].copy()

        # ── 2. Encode target if it's text ─────────────────────────────────────
        label_classes = None
        if y.dtype == "object" or y.dtype.name == "category":
            y = self.target_encoder.fit_transform(y)
            label_classes = self.target_encoder.classes_.tolist()
            print(f"[target]  Encoded labels: {label_classes}")

        # ── 3. BUILD PIPELINE (before split — just inspects dtypes, no fitting) ──
        pipeline, num_cols, cat_cols = self._build_pipeline(X)
        feature_names = num_cols + cat_cols

        # ── 4. SPLIT FIRST — then fit ─────────────────────────────────────────
        #   stratify=y ensures class proportions are preserved in both splits.
        #   Critical for imbalanced datasets (wine imbalance ratio: 68:1).
        stratify_arg = y if problem_type == "classification" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = test_size,
            random_state = random_state,
            stratify     = stratify_arg,
        )
        print(f"[split]  Train: {X_train.shape}  |  Test: {X_test.shape}")

        # ── 5. FIT ON TRAIN ONLY, then transform both ─────────────────────────
        #   The scaler learns mean/std from training rows only.
        #   The test set is transformed using those same train statistics.
        #   This prevents data leakage.
        X_train_arr = pipeline.fit_transform(X_train)
        X_test_arr  = pipeline.transform(X_test)

        # ── 6. Back to DataFrame (preserves column names for feature importance) ─
        X_train_out = pd.DataFrame(X_train_arr, columns=feature_names)
        X_test_out  = pd.DataFrame(X_test_arr,  columns=feature_names)

        self.pipeline = pipeline   # store for optional .pkl export

        print(f"[done]   X_train: {X_train_out.shape}  X_test: {X_test_out.shape}")
        return X_train_out, X_test_out, y_train, y_test, pipeline, feature_names, label_classes

    # ── EXPORT ────────────────────────────────────────────────────────────────

    def save_pipeline(self, out_path: str):
        """Serialise the fitted pipeline to disk (for Layer 4 model runner)."""
        if self.pipeline is None:
            raise RuntimeError("Call process_and_split() before save_pipeline().")
        joblib.dump(self.pipeline, out_path)
        print(f"[export]  Pipeline saved → {out_path}")


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import sys

    # Resolve project root the same way test_main.py does
    # This file lives at: backend/services/preprocessing/tabular_preprocessing.py
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    # Read from session.json — mirrors how Layer 2 will call this in production
    import json
    SESSION_FILE = PROJECT_ROOT / "data" / "sessions" / "session.json"

    if SESSION_FILE.exists():
        session      = json.loads(SESSION_FILE.read_text())
        file_path    = session["dataset_path"] if "dataset_path" in session else \
                       str(PROJECT_ROOT / "data" / "datasets" / session["filename"])
        target_col   = session["target_column"]
        problem_type = session.get("problem_type", "classification")
        test_size    = 1 - session.get("split_ratio", 0.8)
        random_seed  = session.get("random_seed", 42)
        print(f"[session]  Loaded from {SESSION_FILE}")
    else:
        # Fallback for quick local testing without a running backend
        print("[session]  session.json not found — using hardcoded fallback")
        file_path    = str(PROJECT_ROOT / "data" / "datasets" / "wine" / "winequality-red.csv")
        target_col   = "quality"
        problem_type = "classification"
        test_size    = 0.2
        random_seed  = 42

    proc = DynamicPreprocessor(file_path=file_path, target_col=target_col)
    X_train, X_test, y_train, y_test, pipeline, features, classes = proc.process_and_split(
        test_size    = test_size,
        random_state = random_seed,
        problem_type = problem_type,
    )

    print("\n--- X_train (first 3 rows) ---")
    print(X_train.head(3).to_string())

    print(f"\nFeature names : {features}")
    print(f"Label classes : {classes}")

    # Optional: save the pipeline for Layer 4
    pipeline_path = str(PROJECT_ROOT / "data" / "features" / "tabular_pipeline.pkl")
    Path(pipeline_path).parent.mkdir(parents=True, exist_ok=True)
    proc.save_pipeline(pipeline_path)
