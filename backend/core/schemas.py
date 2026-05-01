from pydantic import BaseModel
from typing import Dict, Optional, Any

class DatasetFlags(BaseModel):
    # Common flags
    has_missing: bool
    high_missing: bool
    imbalanced: bool
    large_dataset: bool
    
    # Tabular specific (Defaults to False if missing)
    high_cardinality: Optional[bool] = False
    has_text_features: Optional[bool] = False
    
    # Text specific (Defaults to False if missing)
    has_duplicates: Optional[bool] = False
    high_duplicates: Optional[bool] = False
    many_classes: Optional[bool] = False
    is_short_text: Optional[bool] = False
    is_long_text: Optional[bool] = False
    large_vocab: Optional[bool] = False
    noisy_text: Optional[bool] = False

class DatasetProfile(BaseModel):
    dtype: str          # 'tabular' or 'text'
    n_rows: int
    n_columns: int
    flags: DatasetFlags
    
    # Optional metrics
    n_classes: Optional[int] = None
    imbalance_ratio: Optional[float] = None
    
    # Catch-all for text_length_stats, vocabulary, model_hints, etc.
    # This prevents Pydantic from crashing on keys it doesn't strictly need for Layer 3.
    model_config = {"extra": "allow"}