from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from models.basemodel import BaseModel

class RandomForestCV(BaseModel):
    def __init__(self, 
                 n_estimators=200, 
                 max_depth=15, 
                 min_samples_leaf=5, 
                 min_samples_split=12, 
                 n_iter=5, 
                 max_features=15,
                 scoring='neg_root_mean_squared_error'
                ):
        
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_iter = n_iter
        self.max_features = max_features
        self.scoring = scoring

    def fit(self, X, y, oof_idx=None):
        self.max_features = min(self.max_features, X.shape[1])
        model = RandomForestRegressor(max_features=self.max_features)

        self.model = RandomizedSearchCV(model, 
                                   param_distributions={
                                       'n_estimators': self.n_estimators,
                                       'max_depth': self.max_depth,
                                       'min_samples_leaf': self.min_samples_leaf,
                                       'min_samples_split': self.min_samples_split
                                   },
                                   n_iter=self.n_iter,
                                   scoring=self.scoring,
                                   cv=PredefinedSplit(oof_idx),
                                   refit=True)

        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
            