import numpy as np
from models.basemodel import BaseModel
from autogluon.tabular import TabularPredictor

class AutoGluon(BaseModel):
    def __init__(self, time=60, stack_levels=2, bag_folds=5, bag_sets=5, scoring='root_mean_squared_error', hp_tune=False):
        super().__init__()
        self.time = time
        self.stack_levels = stack_levels
        self.bag_folds = bag_folds
        self.bag_sets = bag_sets
        self.label = 'GPP'   
        self.scoring = scoring
        self.hp_tune = hp_tune
    
    def fit(self, X, y, oof_idx=None):

        # create random id for temp files stored by the package
        ## TODO not ideal, but necessary as path would be the same in parallel computation

        self.id = np.random.RandomState().randint(0, 1e6)
        self.model = TabularPredictor(label=self.label, groups='groups', eval_metric=self.scoring, path='tmp/AutogluonModels/ag-' + str(self.id))

        train = X.copy()
        train[self.label] = y
        train['groups'] = train.index.get_level_values(0).values

        self.model.fit(train, 
              time_limit=self.time,
              num_stack_levels=self.stack_levels,
              num_bag_folds=self.bag_folds,
              #hyperparameter_tune=self.hp_tune,
              refit_full=True,
              #set_best_to_refit_full=True,
              num_bag_sets=self.bag_sets)

    def predict(self, X):
        return self.model.predict(X)