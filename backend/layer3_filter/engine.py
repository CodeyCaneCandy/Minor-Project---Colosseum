import yaml
import rule_engine
from typing import List, Dict, Any

# Assuming your Pydantic schemas are in core/schemas.py
from core.schemas import DatasetProfile 

# 1. IMPORT YOUR NEW LOGGER HERE
from layer3_filter.logger import FilterStreamLogger

class RuleLogger:
    """A simple placeholder logger. Later, this will push WebSockets to your UI."""
    def log(self, action: str, target: str, reason: str):
        print(f"[{action.upper()}] {target} -> {reason}")

class DatasetModelFilter:
    def __init__(self, rules_path: str):
        self.rules = self._load_rules(rules_path)
        self.logger = FilterStreamLogger()
        
        # Colosseum's Master List of supported models
        self.master_models = [
            "SVM", "LogReg", "NaiveBayes", "KNN", 
            "DecisionTree", "RandomForest", "XGBoost"
        ]

    def _load_rules(self, path: str) -> List[Dict]:
        """Loads the YAML file safely."""
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
            return data.get('rules', [])

    # 1. ADD 'async' HERE
    async def apply_rules(self, profile: DatasetProfile) -> Dict[str, Any]:
        """Runs the dataset profile through the YAML gauntlet."""
        
        profile_dict = profile.model_dump() 
        active_models = set(self.master_models)
        warnings = []
        restrictions = []

        for rule in self.rules:
            try:
                engine = rule_engine.Rule(rule['condition']) 
            except rule_engine.errors.RuleSyntaxError as e:
                print(f"Syntax error in rule {rule['id']}: {e}")
                continue

            if engine.matches(profile_dict):
                action = rule['action'].lower()
                targets = rule['target']
                reason = rule['reason']

                if "All" in targets:
                    targets = list(active_models)

                for target in targets:
                    if action == "exclude" and target in active_models:
                        active_models.remove(target)
                        # 2. ADD 'await' HERE
                        await self.logger.log_event(action, target, reason)
                        
                    elif action == "warn":
                        warnings.append({"model": target, "reason": reason})
                        # 3. ADD 'await' HERE
                        await self.logger.log_event(action, target, reason)
                        
                    elif action == "restrict":
                        restrictions.append({"model": target, "reason": reason})
                        # 4. ADD 'await' HERE
                        await self.logger.log_event(action, target, reason)

        return {
            "ready_models": list(active_models),
            "warnings": warnings,
            "restrictions": restrictions
        }