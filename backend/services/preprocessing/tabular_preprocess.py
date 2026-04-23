import csv
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Split is its own module — import from there
from services.preprocessing.train_test_split import split_dataset


class DynamicPreprocessor:
    """
    Layer 2 — Tabular feature engineering pipeline.

    Responsibility: load → encode target → build sklearn pipeline
                    → fit on train → transform train and test.

    Splitting is delegated to train_test_split.py so the same
    split logic can be reused by text and image preprocessors.

    Call order
    ----------
    preprocessor = DynamicPreprocessor(file_path, target_col)
    result       = preprocessor.run(session)
    # result keys: X_train, X_test, y_train, y_test,
    #              pipeline, feature_names, label_classes
    """

    def __init__(self, file_path: str, target_col: str):
        self.file_path      = file_path
        self.target_col     = target_col
        self.target_encoder = LabelEncoder()
        self.pipeline       = None

    # ── LOAD ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_delimiter(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return ","

    def load_data(self) -> pd.DataFrame:
        sep = self._detect_delimiter(self.file_path)
        print(f"[load]   delimiter='{sep}'")
        df  = pd.read_csv(self.file_path, sep=sep)
        df.columns = df.columns.str.strip()

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found.\n"
                f"Available: {df.columns.tolist()}"
            )
        print(f"[load]   shape={df.shape}")
        return df

    # ── PIPELINE ─────────────────────────────────────────────────────────────

    def _build_pipeline(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        print(f"[pipeline]  numerical={len(num_cols)}  categorical={len(cat_cols)}")

        transformers = []

        if num_cols:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), num_cols))

        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]), cat_cols))

        pipeline      = ColumnTransformer(transformers=transformers)
        feature_names = num_cols + cat_cols
        return pipeline, feature_names

    # ── MAIN ENTRY ────────────────────────────────────────────────────────────

    def run(self, session: dict) -> dict:
        """
        Full pipeline driven by session dict.

        Reads: session["split_ratio"], session["random_seed"],
               session["problem_type"]

        Returns dict with: X_train, X_test, y_train, y_test,
                           pipeline, feature_names, label_classes
        """
        train_split  = session.get("split_ratio",  0.8)
        random_state = session.get("random_seed",   42)
        problem_type = session.get("problem_type", "classification")

        df = self.load_data()

        # ── 1. Encode target ─────────────────────────────────────────────────
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].copy()

        label_classes = None
        if y.dtype == "object" or y.dtype.name == "category":
            y_encoded     = self.target_encoder.fit_transform(y)
            label_classes = self.target_encoder.classes_.tolist()
            print(f"[target]  encoded classes={label_classes}")
        else:
            y_encoded = y.values

        # ── 2. Split BEFORE fitting (prevents data leakage) ──────────────────
        # We temporarily put y back into df just for the split call,
        # then separate again so the pipeline only sees X.
        df_with_y = X.copy()
        df_with_y["__target__"] = y_encoded

        split = split_dataset(
            df          = df_with_y,
            target_column = "__target__",
            train_split   = train_split,
            random_state  = random_state,
            problem_type  = problem_type,
        )

        X_train_raw = split["X_train"]
        X_test_raw  = split["X_test"]
        y_train     = split["y_train"]
        y_test      = split["y_test"]

        # ── 3. Fit pipeline on TRAIN only ────────────────────────────────────
        pipeline, feature_names = self._build_pipeline(X_train_raw)

        X_train_arr = pipeline.fit_transform(X_train_raw)
        X_test_arr  = pipeline.transform(X_test_raw)

        # Back to DataFrame — preserves column names for feature importance
        X_train_out = pd.DataFrame(X_train_arr, columns=feature_names)
        X_test_out  = pd.DataFrame(X_test_arr,  columns=feature_names)

        self.pipeline = pipeline
        print(f"[done]   X_train={X_train_out.shape}  X_test={X_test_out.shape}")

        return {
            "X_train":       X_train_out,
            "X_test":        X_test_out,
            "y_train":       y_train,
            "y_test":        y_test,
            "pipeline":      pipeline,
            "feature_names": feature_names,
            "label_classes": label_classes,
        }

    # ── EXPORT ────────────────────────────────────────────────────────────────

    def save_pipeline(self, out_path: str):
        if self.pipeline is None:
            raise RuntimeError("Call run() before save_pipeline().")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out_path)
        print(f"[export]  pipeline → {out_path}")


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    # Re-import with correct root on path
    from backend.services.preprocessing.train_test_split import split_dataset  # noqa

    SESSION_FILE = PROJECT_ROOT / "data" / "sessions" / "session.json"

    if SESSION_FILE.exists():
        session = json.loads(SESSION_FILE.read_text())
        file_path   = session.get("dataset_path") or \
                      str(PROJECT_ROOT / "data" / "datasets" / session["filename"])
        target_col  = session["target_column"]
        print(f"[session]  loaded from {SESSION_FILE}")
    else:
        print("[session]  no session.json found — using fallback")
        file_path  = str(PROJECT_ROOT / "data" / "datasets" / "wine" / "winequality-red.csv")
        target_col = "quality"
        session    = {
            "split_ratio":  0.8,
            "random_seed":  42,
            "problem_type": "classification",
        }

    proc   = DynamicPreprocessor(file_path=file_path, target_col=target_col)
    result = proc.run(session)

    print("\n--- X_train (first 3 rows) ---")
    print(result["X_train"].head(3).to_string())
    print(f"\nfeature_names : {result['feature_names']}")
    print(f"label_classes : {result['label_classes']}")

    proc.save_pipeline(
        str(PROJECT_ROOT / "data" / "features" / "tabular_pipeline.pkl")
    )
