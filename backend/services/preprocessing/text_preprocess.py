import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# Reusing your excellent modular split utility
from services.preprocessing.train_test_split import split_dataset

class TextPreprocessor:
    """
    Layer 2 — Text feature engineering pipeline.
    
    Handles two paths based on the session config or profiler hints:
    - Path A: TF-IDF (Sparse, fast, good for medium/long text)
    - Path B: Sentence-Transformers (Dense, semantic, good for short/complex text)
    """

    def __init__(self, file_path: str, text_col: str, target_col: str = None):
        self.file_path      = file_path
        self.text_col       = text_col
        self.target_col     = target_col
        self.target_encoder = LabelEncoder()
        
        # Pipeline state
        self.method         = "tfidf" # Default, overridden in run()
        self.vectorizer     = None
        self.embedder       = None

    def load_data(self) -> pd.DataFrame:
        """Loads data, assuming CSV for now based on profiler logic."""
        # Note: In production, you'd reuse your delimiter detection here
        df = pd.read_csv(self.file_path)
        df.columns = df.columns.str.strip()

        if self.text_col not in df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found.")
        
        # Drop rows where the text is NaN
        df = df.dropna(subset=[self.text_col]).reset_index(drop=True)
        return df

    def _apply_tfidf(self, X_train: pd.Series, X_test: pd.Series):
        """Path A: TF-IDF Vectorization"""
        print("[pipeline]  Applying TF-IDF Vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=1500, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit on train, transform both
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec  = self.vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec

    def _apply_embeddings(self, X_train: pd.Series, X_test: pd.Series):
        """Path B: Sentence Transformers (Dense Embeddings)"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[pipeline]  Applying all-MiniLM-L6-v2 embeddings on {device.upper()}...")
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Batch encode to save memory
        X_train_vec = self.embedder.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
        X_test_vec  = self.embedder.encode(X_test.tolist(), batch_size=32, show_progress_bar=True)
        
        return X_train_vec, X_test_vec

    def run(self, session: dict) -> dict:
        """Executes the text pipeline driven by the session dict."""
        train_split  = session.get("split_ratio", 0.8)
        random_state = session.get("random_seed", 42)
        problem_type = session.get("problem_type", "classification")
        
        # Check if the UI/Profiler forced an embedding method, default to tfidf
        self.method  = session.get("text_representation", "tfidf") 

        df = self.load_data()

        # ── 1. Encode target ─────────────────────────────────────────────────
        label_classes = None
        if self.target_col and self.target_col in df.columns:
            y = df[self.target_col].copy()
            if y.dtype == "object" or y.dtype.name == "category":
                y_encoded     = self.target_encoder.fit_transform(y)
                label_classes = self.target_encoder.classes_.tolist()
            else:
                y_encoded = y.values
            
            df["__target__"] = y_encoded
            target_for_split = "__target__"
        else:
            y_encoded = None
            target_for_split = None

        # ── 2. Split dataset ─────────────────────────────────────────────────
        split = split_dataset(
            df            = df,
            target_column = target_for_split,
            train_split   = train_split,
            random_state  = random_state,
            problem_type  = problem_type,
        )

        X_train_raw = split["X_train"][self.text_col].astype(str)
        X_test_raw  = split["X_test"][self.text_col].astype(str)
        y_train     = split["y_train"]
        y_test      = split["y_test"]

        # ── 3. Vectorize Text ────────────────────────────────────────────────
        if self.method == "embeddings":
            X_train_arr, X_test_arr = self._apply_embeddings(X_train_raw, X_test_raw)
            feature_names = [f"dim_{i}" for i in range(X_train_arr.shape[1])]
        else:
            X_train_arr, X_test_arr = self._apply_tfidf(X_train_raw, X_test_raw)
            feature_names = self.vectorizer.get_feature_names_out().tolist()

        print(f"[done]   X_train={X_train_arr.shape}  X_test={X_test_arr.shape}")

        return {
            "X_train":       X_train_arr,
            "X_test":        X_test_arr,
            "y_train":       y_train,
            "y_test":        y_test,
            "method":        self.method,
            "feature_names": feature_names,
            "label_classes": label_classes,
        }

    def save_pipeline(self, out_path: str):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # We only save the TF-IDF vectorizer. SentenceTransformers are pre-trained weights
        # that we just load by name during inference, so no need to pickle them here.
        if self.method == "tfidf" and self.vectorizer:
            joblib.dump(self.vectorizer, out_path)
            print(f"[export]  TF-IDF vectorizer → {out_path}")