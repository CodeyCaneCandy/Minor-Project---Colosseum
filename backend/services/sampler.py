import pandas as pd
from sklearn.model_selection import train_test_split

def smart_sample(df: pd.DataFrame, target_col: str, max_rows: int = 15000) -> tuple[pd.DataFrame, dict]:
    """
    Trims massive datasets down to a manageable size using stratified sampling
    so the models train fast without losing proportional class representation.
    """
    total_rows = len(df)
    
    # If it's already small enough, just let it pass through
    if total_rows <= max_rows:
        return df, {
            "sampling_applied": False,
            "original_rows": total_rows,
            "sampled_rows": total_rows,
            "sample_fraction": 1.0,
            "reason": f"Dataset is under the {max_rows}-row threshold."
        }

    print(f"\n[LAYER 2.5] Dataset too large ({total_rows} rows). Trimming to {max_rows} rows...")
    
    try:
        # Perform a stratified split to maintain class balance
        sampled_df, _ = train_test_split(
            df,
            train_size=max_rows,
            stratify=df[target_col],
            random_state=42
        )
    except ValueError:
        # Fallback if stratify fails (e.g., for regression tasks with continuous targets)
        sampled_df = df.sample(n=max_rows, random_state=42)

    report = {
        "sampling_applied": True,
        "original_rows": total_rows,
        "sampled_rows": max_rows,
        "sample_fraction": round(max_rows / total_rows, 3),
        "reason": f"Capped at {max_rows:,} rows to ensure rapid model evaluation."
    }
    
    return sampled_df, report