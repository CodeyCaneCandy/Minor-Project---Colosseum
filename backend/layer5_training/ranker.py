class WeightedRanker:
    def __init__(self, task_type="classification"):
        self.task_type = task_type
        
        # Ideally, these come from your config/settings.yaml later!
        self.clf_weights = {"f1": 0.45, "roc_auc": 0.35, "accuracy": 0.20}
        self.reg_weights = {"r2": 0.70, "rmse": -0.30} # RMSE is negative because lower is better

    def score_models(self, final_models: dict) -> list:
        """Applies the weighted composite score to all models and sorts them."""
        for name, data in final_models.items():
            metrics = data["metrics"]
            
            if self.task_type == "classification":
                score = (
                    (metrics["f1"] * self.clf_weights["f1"]) + 
                    (metrics["roc_auc"] * self.clf_weights["roc_auc"]) + 
                    (metrics["accuracy"] * self.clf_weights["accuracy"])
                )
            else:
                # Basic regression composite logic
                score = metrics["r2"] # (Will expand this when regression is fully implemented)

            # Penalize slightly for massive size or slow inference (Efficiency metrics)
            efficiency = data["efficiency"]
            size_penalty = min(efficiency["size_mb"] * 0.001, 0.05) # Max 5% penalty
            
            data["composite_score"] = round(score - size_penalty, 4)

        # Return list of tuples: [('RandomForest', {...}), ('SVM', {...})]
        return sorted(final_models.items(), key=lambda x: x[1]["composite_score"], reverse=True)

    def evaluate_confidence(self, ranked_models: list) -> dict:
        """Calculates C = (S1 - S2) / S1 * 100"""
        if len(ranked_models) < 2:
            return {"level": "high", "value": 100.0, "gap": 0}

        s1 = ranked_models[0][1]["composite_score"]
        s2 = ranked_models[1][1]["composite_score"]
        
        if s1 <= 0:
            return {"level": "low", "value": 0.0, "gap": 0}

        c_value = round(((s1 - s2) / s1) * 100, 1)

        if c_value > 15.0:
            level = "high"
        elif c_value >= 5.0:
            level = "medium"
        else:
            level = "low"

        return {"level": level, "value": c_value}

    def generate_report(self, ranked_models: list, classes: list) -> dict:
        """Assembles the final JSON payload for the frontend."""
        winner_name = ranked_models[0][0]
        winner_data = ranked_models[0][1]
        
        confidence = self.evaluate_confidence(ranked_models)
        
        # Build the dynamic explanation
        if confidence["level"] == "high":
            explanation = f"{winner_name} was the clear winner, outperforming the runner-up ({ranked_models[1][0]}) by a definitive margin of {confidence['value']}%."
        elif confidence["level"] == "medium":
            explanation = f"{winner_name} narrowly beat {ranked_models[1][0]}. Both models performed well, achieving high overall stability."
        else:
            explanation = f"It was a statistical tie between {winner_name} and {ranked_models[1][0]} (Score gap: {confidence['value']}%). You should check deployment efficiency metrics to make a final decision."

        # Re-pack the models dict cleanly for the UI
        clean_models = {}
        for name, data in ranked_models:
            clean_models[name] = {
                "metrics": data["metrics"],
                "efficiency": data["efficiency"],
                "composite_score": data["composite_score"]
            }

        return {
            "models": clean_models,
            "winner": {"name": winner_name, "score": winner_data["composite_score"]},
            "confidence": confidence,
            "explanation": explanation,
            "confusion_matrix": winner_data.get("cm"),
            "label_classes": classes
        }
        