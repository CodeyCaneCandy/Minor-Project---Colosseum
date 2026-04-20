import csv
import pandas as pd
import numpy as np
from pathlib import Path


def _detect_separator(file_path: str) -> str:
    """Read the first 4KB and let csv.Sniffer detect the delimiter."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","   # safe default


def profile_tabular(file_path, target_column, problem_type):

    # ── AUTO-DETECT SEPARATOR ────────────────────────────────────────────────
    sep = _detect_separator(file_path)
    df  = pd.read_csv(file_path, sep=sep)

    # Strip any stray whitespace from column names (common in CSVs)
    df.columns = df.columns.str.strip()

    # ── GUARD: check target column exists ───────────────────────────────────
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    profile = {}

    # =========================
    # 1. BASIC INFO
    # =========================
    profile["n_rows"]    = len(df)
    profile["n_columns"] = df.shape[1]
    profile["separator"] = sep   # useful to log for debugging

    features = df.columns.tolist()
    features.remove(target_column)

    # =========================
    # 2. FEATURE TYPES
    # =========================
    num_cols = df[features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = df[features].select_dtypes(exclude=np.number).columns.tolist()

    # Detect possible TEXT columns (avg string length heuristic)
    text_cols = []
    for col in cat_cols:
        avg_len = df[col].dropna().astype(str).str.len().mean()
        if avg_len > 20:
            text_cols.append(col)

    cat_cols = [c for c in cat_cols if c not in text_cols]

    profile["feature_types"] = {
        "numerical":   num_cols,
        "categorical": cat_cols,
        "text_like":   text_cols,
    }

    # =========================
    # 3. NULL STATS
    # =========================
    null_percent = (df.isnull().mean() * 100).round(2).to_dict()
    profile["null_percent"] = null_percent

    # =========================
    # 4. CARDINALITY
    # =========================
    cardinality = {col: int(df[col].nunique()) for col in cat_cols}
    profile["categorical_cardinality"] = cardinality

    # =========================
    # 5. TARGET ANALYSIS
    # =========================
    if problem_type == "classification":
        class_counts = df[target_column].value_counts().to_dict()
        profile["class_distribution"] = class_counts

        max_c = max(class_counts.values())
        min_c = max(min(class_counts.values()), 1)   # guard against div-by-zero

        profile["imbalance_ratio"] = round(max_c / min_c, 2)
        profile["n_classes"]       = len(class_counts)

    elif problem_type == "regression":
        profile["target_stats"] = {
            "mean": round(float(df[target_column].mean()), 4),
            "std":  round(float(df[target_column].std()),  4),
            "min":  round(float(df[target_column].min()),  4),
            "max":  round(float(df[target_column].max()),  4),
        }

    # =========================
    # 6. FLAGS (FOR LAYER 3)
    # =========================
    profile["flags"] = {
        "has_missing":      any(v > 0  for v in null_percent.values()),
        "high_missing":     any(v > 30 for v in null_percent.values()),
        "imbalanced":       profile.get("imbalance_ratio", 1) > 3,
        "high_cardinality": any(v > 50 for v in cardinality.values()),
        "large_dataset":    profile["n_rows"] > 50000,
        "has_text_features": len(text_cols) > 0,
    }

    # =========================
    # 7. SUMMARY (FOR LLM)
    # =========================
    profile["summary"] = (
        f"{profile['n_rows']} rows, "
        f"{len(num_cols)} numerical, "
        f"{len(cat_cols)} categorical features. "
        f"Imbalance ratio: {profile.get('imbalance_ratio', 'N/A')}."
    )

    return profile
