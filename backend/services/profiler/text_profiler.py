import csv
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


# ── OPTIONAL: language detection ────────────────────────────────────────────
# langdetect is a lightweight library (~1MB). If not installed, language
# detection is skipped gracefully — everything else still works.
try:
    from langdetect import detect as _detect_lang
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0          # makes detection deterministic
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


# ── INTERNALS ────────────────────────────────────────────────────────────────

def _detect_separator(file_path: str) -> str:
    """Sniff CSV delimiter from the first 4KB."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _token_count(text: str) -> int:
    """Naive whitespace tokeniser — no NLTK dependency needed."""
    return len(str(text).split())


def _detect_language(series: pd.Series, sample_n: int = 50) -> str:
    """
    Sample up to sample_n rows and vote on the most common detected language.
    Returns ISO 639-1 code (e.g. 'en', 'fr') or 'unknown'.
    """
    if not _LANGDETECT_AVAILABLE:
        return "unknown (langdetect not installed)"

    sample = series.dropna().astype(str).sample(
        min(sample_n, len(series)), random_state=42
    )
    detected = []
    for text in sample:
        try:
            detected.append(_detect_lang(text))
        except Exception:
            pass

    if not detected:
        return "unknown"

    return Counter(detected).most_common(1)[0][0]


def _vocab_size(series: pd.Series, sample_n: int = 5000) -> int:
    """Unique lowercase tokens across a sample of the corpus."""
    sample = series.dropna().astype(str)
    if len(sample) > sample_n:
        sample = sample.sample(sample_n, random_state=42)

    words = " ".join(sample).lower()
    words = re.sub(r"[^a-z0-9\s]", " ", words)   # strip punctuation
    return len(set(words.split()))


def _avg_special_char_ratio(series: pd.Series) -> float:
    """Fraction of characters that are non-alphanumeric (URLs, hashtags, etc.)"""
    def ratio(text):
        text = str(text)
        if len(text) == 0:
            return 0.0
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special / len(text)

    return round(float(series.dropna().apply(ratio).mean()), 4)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def profile_text(file_path: str, text_column: str, target_column: str = None):
    """
    Profile a text classification dataset.

    Parameters
    ----------
    file_path     : path to the CSV file
    text_column   : name of the column containing raw text
    target_column : name of the label column (optional — omit for unsupervised)

    Returns
    -------
    dict — profile consumed by Layer 3 filter and Layer 6 LLM explainer
    """

    # ── LOAD ─────────────────────────────────────────────────────────────────
    sep = _detect_separator(file_path)
    df  = pd.read_csv(file_path, sep=sep)
    df.columns = df.columns.str.strip()

    # ── GUARD: columns exist ─────────────────────────────────────────────────
    missing = []
    if text_column not in df.columns:
        missing.append(f"text_column='{text_column}'")
    if target_column and target_column not in df.columns:
        missing.append(f"target_column='{target_column}'")
    if missing:
        raise ValueError(
            f"Column(s) not found: {', '.join(missing)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    profile = {}

    # =========================================================================
    # 1. BASIC INFO
    # =========================================================================
    profile["dtype"]       = "text"
    profile["n_rows"]       = len(df)
    profile["n_columns"]    = df.shape[1]
    profile["text_column"]  = text_column
    profile["target_column"] = target_column
    profile["separator"]    = sep

    text_series = df[text_column].astype(str)

    # =========================================================================
    # 2. TEXT LENGTH STATS
    # =========================================================================
    char_lengths  = text_series.str.len()
    token_counts  = text_series.apply(_token_count)

    profile["text_length_stats"] = {
        "mean_chars":   round(float(char_lengths.mean()),  2),
        "median_chars": round(float(char_lengths.median()), 2),
        "max_chars":    int(char_lengths.max()),
        "min_chars":    int(char_lengths.min()),
        "std_chars":    round(float(char_lengths.std()),   2),

        "mean_tokens":   round(float(token_counts.mean()),  2),
        "median_tokens": round(float(token_counts.median()), 2),
        "max_tokens":    int(token_counts.max()),
        "min_tokens":    int(token_counts.min()),
    }

    # =========================================================================
    # 3. TEXT QUALITY
    # =========================================================================
    null_count  = int(df[text_column].isnull().sum())
    empty_count = int((text_series.str.strip() == "").sum())
    dup_count   = int(text_series.duplicated().sum())

    profile["text_quality"] = {
        "null_count":       null_count,
        "null_percent":     round(null_count / len(df) * 100, 2),
        "empty_count":      empty_count,
        "duplicate_count":  dup_count,
        "duplicate_percent": round(dup_count / len(df) * 100, 2),
        "special_char_ratio": _avg_special_char_ratio(text_series),
    }

    # =========================================================================
    # 4. VOCABULARY
    # =========================================================================
    profile["vocabulary"] = {
        "vocab_size":       _vocab_size(text_series),
        "language":         _detect_language(text_series),
    }

    # =========================================================================
    # 5. NULL STATS (all columns — mirrors tabular profiler format)
    # =========================================================================
    profile["null_percent"] = (df.isnull().mean() * 100).round(2).to_dict()

    # =========================================================================
    # 6. TARGET / LABEL ANALYSIS
    # =========================================================================
    if target_column:
        class_counts = df[target_column].value_counts().to_dict()
        profile["class_distribution"] = class_counts

        max_c = max(class_counts.values())
        min_c = max(min(class_counts.values()), 1)   # guard div-by-zero

        profile["imbalance_ratio"] = round(max_c / min_c, 2)
        profile["n_classes"]       = len(class_counts)
    else:
        profile["class_distribution"] = None
        profile["imbalance_ratio"]    = None
        profile["n_classes"]          = None

    # =========================================================================
    # 7. TEXT TYPE HEURISTICS
    # =========================================================================
    mean_tokens = profile["text_length_stats"]["mean_tokens"]

    if mean_tokens < 15:
        text_type = "short_text"       # tweets, titles, search queries
    elif mean_tokens < 100:
        text_type = "medium_text"      # reviews, comments, paragraphs
    else:
        text_type = "long_text"        # articles, essays, documents

    profile["text_type"] = text_type

    # =========================================================================
    # 8. FLAGS (FOR LAYER 3 FILTER)
    # =========================================================================
    profile["flags"] = {
        # Data quality
        "has_missing":       null_count > 0,
        "high_missing":      profile["text_quality"]["null_percent"] > 10,
        "has_duplicates":    dup_count > 0,
        "high_duplicates":   profile["text_quality"]["duplicate_percent"] > 15,

        # Label
        "imbalanced":        (profile["imbalance_ratio"] or 1) > 3,
        "many_classes":      (profile["n_classes"] or 0) > 10,

        # Text characteristics
        "is_short_text":     text_type == "short_text",
        "is_long_text":      text_type == "long_text",
        "large_vocab":       profile["vocabulary"]["vocab_size"] > 50000,
        "large_dataset":     profile["n_rows"] > 50000,
        "noisy_text":        profile["text_quality"]["special_char_ratio"] > 0.15,
    }

    # =========================================================================
    # 9. MODEL HINTS (PASSED TO LAYER 3)
    # =========================================================================
    # Pre-computed hints so the rule engine doesn't need to re-derive them
    profile["model_hints"] = {
        # TF-IDF works best on medium+ text with decent vocabulary
        "tfidf_suitable": mean_tokens >= 5 and profile["vocabulary"]["vocab_size"] > 500,

        # Embeddings shine on short text or when semantics matter more than frequency
        "embeddings_recommended": text_type == "short_text" or mean_tokens < 30,

        # Flag for Layer 3: huge vocab → warn about TF-IDF memory
        "warn_tfidf_memory": profile["vocabulary"]["vocab_size"] > 100000,
    }

    # =========================================================================
    # 10. SUMMARY (FOR LLM EXPLAINER)
    # =========================================================================
    lang    = profile["vocabulary"]["language"]
    n_cls   = profile["n_classes"] or "N/A"
    imb     = profile["imbalance_ratio"] or "N/A"
    ttype   = text_type.replace("_", " ")

    profile["summary"] = (
        f"{profile['n_rows']} rows of {ttype} "
        f"(avg {mean_tokens:.0f} tokens/doc), "
        f"vocab size {profile['vocabulary']['vocab_size']:,}, "
        f"language: {lang}. "
        f"{n_cls} classes, imbalance ratio: {imb}."
    )

    return profile
