from sklearn.ensemble import ExtraTreesClassifier

class UltraRandomTree:
    """A custom algorithm dropped in by the user."""
    def __init__(self):
        self.model = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42)
        
    def fit(self, X, y):
        # The runner requires a .fit() method
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        # The runner requires a .predict() method
        return self.model.predict(X)