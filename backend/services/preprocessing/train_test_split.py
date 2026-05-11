import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split


def split_dataset(
    df:             pd.DataFrame,
    target_column:  str   = None,
    train_split:    float = 0.8,
    random_state:   int   = 42,
    problem_type:   str   = "classification",
) -> dict:
    """
    Stratified train/test split on a DataFrame.

    Kept as a standalone utility so tabular_preprocessing.py,
    text_preprocessing.py, and any future preprocessor can all
    import the same split logic without duplicating it.

    Parameters
    ----------
    df            : cleaned, encoded DataFrame (post-preprocessing)
    target_column : label column name. If None, splits X only (unsupervised).
    train_split   : fraction for training  (e.g. 0.8 → 80/20 split)
    random_state  : seed for reproducibility
    problem_type  : "classification" → stratify=y
                    "regression"     → no stratification (continuous targets)

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test
    y_train / y_test are None when target_column is not provided.
    """

    test_size = round(1 - train_split, 4)

    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None

    # Stratify only for classification — continuous targets crash stratify
    stratify_arg = y if (y is not None and problem_type == "classification") else None

    if y is not None:
        X_train, X_test, y_train, y_test = _sklearn_split(
            X, y,
            test_size    = test_size,
            random_state = random_state,
            stratify     = stratify_arg,
        )
    else:
        X_train, X_test = _sklearn_split(
            X,
            test_size    = test_size,
            random_state = random_state,
        )
        y_train = y_test = None

    print(
        f"[split]  train={len(X_train)}  test={len(X_test)}"
        f"  stratified={'yes' if stratify_arg is not None else 'no'}"
    )

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    dummy = pd.DataFrame({
        "feat_a": np.random.randn(200),
        "feat_b": np.random.randn(200),
        "label":  (["cat", "dog"] * 100),
    })
    result = split_dataset(dummy, target_column="label", problem_type="classification")
    print("X_train shape:", result["X_train"].shape)
    print("X_test  shape:", result["X_test"].shape)
    print("y_train dist :", result["y_train"].value_counts().to_dict())
