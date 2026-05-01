# --- QUICK TEST ---
if __name__ == "__main__":
    from core.schemas import DatasetProfile, DatasetFlags
    
    # 1. Mock the data exactly as your Tabular Profiler outputs it
    mock_dataset = DatasetProfile(
        dtype="tabular",
        n_rows=1599,
        n_columns=12,
        n_classes=6,
        imbalance_ratio=68.1,
        flags=DatasetFlags(
            has_missing=False,
            high_missing=False,
            imbalanced=True,         # <--- This should trigger a warning
            large_dataset=False,
            high_cardinality=False,
            has_text_features=False
        )
    )

    # 2. Run the engine (make sure the path matches where you saved your YAML)
    print("\n--- Starting Colosseum Model Filter ---")
    filter_engine = DatasetModelFilter("rules/master_rules.yaml")
    results = filter_engine.apply_rules(mock_dataset)
    
    print("\n--- Final Results ---")
    print(f"Models Ready for Training: {results['ready_models']}")