import csv
import pandas as pd
from pathlib import Path


# ── THRESHOLDS ────────────────────────────────────────────────────────────────
# These mirror the flags already set by the tabular/text profiler.
# If the dataset exceeds these limits, we sample it down before preprocessing.

TABULAR_ROW_LIMIT  = 50_000     # rows
TEXT_ROW_LIMIT     = 50_000     # rows
IMAGE_COUNT_LIMIT  = 5_000      # images per class


def _detect_delimiter(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        import csv as _csv
        dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def apply_sampling_gate(
    file_path:     str,
    target_column: str,
    task_type:     str   = "tabular",
    max_rows:      int   = None,
    random_seed:   int   = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a CSV dataset and apply stratified sampling if it exceeds the row limit.

    Parameters
    ----------
    file_path     : path to the raw CSV file
    target_column : label column (used for stratified sampling)
    task_type     : "tabular" or "text" — determines the row limit if max_rows is None
    max_rows      : override the default row limit (optional)
    random_seed   : for reproducibility

    Returns
    -------
    df      : DataFrame — either the full dataset or a stratified sample
    report  : dict describing what happened (passed into session / profile)

    Example report
    --------------
    {
        "sampling_applied": True,
        "original_rows": 120000,
        "sampled_rows": 50000,
        "sample_fraction": 0.417,
        "reason": "Dataset exceeded 50,000 row limit for tabular data.",
        "strategy": "stratified"
    }
    """

    # ── Determine limit ───────────────────────────────────────────────────────
    if max_rows is None:
        max_rows = TABULAR_ROW_LIMIT if task_type == "tabular" else TEXT_ROW_LIMIT

    # ── Load ──────────────────────────────────────────────────────────────────
    sep = _detect_delimiter(file_path)
    df  = pd.read_csv(file_path, sep=sep)
    df.columns = df.columns.str.strip()

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\n"
            f"Available: {df.columns.tolist()}"
        )

    original_rows = len(df)

    # ── No sampling needed ────────────────────────────────────────────────────
    if original_rows <= max_rows:
        report = {
            "sampling_applied": False,
            "original_rows":    original_rows,
            "sampled_rows":     original_rows,
            "sample_fraction":  1.0,
            "reason":           "Dataset is within the row limit. Full dataset used.",
            "strategy":         "none",
        }
        print(f"[sampling_gate]  No sampling needed ({original_rows:,} rows <= {max_rows:,} limit)")
        return df, report

    # ── Stratified sampling ───────────────────────────────────────────────────
    #
    # We sample proportionally per class so the class distribution is preserved.
    # This is safer than random sampling, especially for imbalanced datasets.
    #
    # sklearn's train_test_split with stratify is the cleanest way to do this:
    # we "split off" max_rows rows and discard the rest.

    from sklearn.model_selection import train_test_split

    sample_fraction = max_rows / original_rows

    # Edge case: a class might have too few samples to stratify.
    # Fall back to random sampling if any class has fewer than 2 members.
    min_class_count = df[target_column].value_counts().min()

    if min_class_count < 2:
        df_sampled = df.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)
        strategy   = "random (stratify not possible — minority class has < 2 samples)"
    else:
        df_sampled, _ = train_test_split(
            df,
            train_size   = max_rows,
            stratify     = df[target_column],
            random_state = random_seed,
        )
        df_sampled = df_sampled.reset_index(drop=True)
        strategy   = "stratified"

    report = {
        "sampling_applied": True,
        "original_rows":    original_rows,
        "sampled_rows":     len(df_sampled),
        "sample_fraction":  round(sample_fraction, 4),
        "reason":           (
            f"Dataset exceeded {max_rows:,} row limit for {task_type} data. "
            f"Stratified sample taken to preserve class distribution."
        ),
        "strategy": strategy,
    }

    print(
        f"[sampling_gate]  Sampling applied: "
        f"{original_rows:,} → {len(df_sampled):,} rows  "
        f"({sample_fraction:.1%} kept)  strategy={strategy}"
    )

    # Show class distribution before and after so the user can verify
    before = df[target_column].value_counts(normalize=True).round(3).to_dict()
    after  = df_sampled[target_column].value_counts(normalize=True).round(3).to_dict()
    report["class_distribution_before"] = before
    report["class_distribution_after"]  = after

    return df_sampled, report


# ── STANDALONE TEST ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import sys, json

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    SESSION_FILE = PROJECT_ROOT / "data" / "sessions" / "session.json"

    if SESSION_FILE.exists():
        session      = json.loads(SESSION_FILE.read_text())
        file_path    = session.get("dataset_path") or \
                       str(PROJECT_ROOT / "data" / "datasets" / session["filename"])
        target_col   = session["target_column"]
        task_type    = session.get("file_type", "tabular")
        random_seed  = session.get("random_seed", 42)
        print(f"[session]  Loaded from {SESSION_FILE}")
    else:
        print("[session]  session.json not found — using fallback")
        file_path   = str(PROJECT_ROOT / "data" / "datasets" / "wine" / "winequality-red.csv")
        target_col  = "quality"
        task_type   = "tabular"
        random_seed = 42

    # Test with an artificially low limit to force sampling on the wine dataset
    df_sampled, report = apply_sampling_gate(
        file_path     = file_path,
        target_column = target_col,
        task_type     = task_type,
        max_rows      = 800,        # wine has 1599 rows — this forces sampling
        random_seed   = random_seed,
    )

    print("\n── Sampling Report ──────────────────────────────────")
    print(json.dumps(report, indent=2))
    print(f"\nReturned DataFrame shape: {df_sampled.shape}")
    print(f"Class distribution after:\n{df_sampled[target_col].value_counts()}")
