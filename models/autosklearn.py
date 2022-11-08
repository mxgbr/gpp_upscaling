from sklearn.model_selection import PredefinedSplit
import autosklearn.regression
from models.basemodel import BaseModel
import numpy as np

class AutoSklearn(BaseModel):
    def __init__(self, 
                 time, 
                 task_time_limit=None, 
                 scoring='mean_squared_error',
                 memory_limit=3072,
                 n_cpus=None
                ):
        
        super().__init__()
        self.time = time
        self.task_time_limit = task_time_limit

        if scoring == 'mean_squared_error':
            self.scoring = autosklearn.metrics.mean_squared_error
        else:
            ## TODO: raise error
            pass

        self.n_cpus = n_cpus
        self.memory_limit = memory_limit

    def fit(self, X, y, oof_idx=None):
        self.id = np.random.RandomState().randint(0, 1e6)
        self.model = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=self.time,
            per_run_time_limit=self.task_time_limit,
            tmp_folder='/tmp/autosklearn_regression_' + str(self.id),
            resampling_strategy=PredefinedSplit(oof_idx),
            metric=self.scoring,
            n_jobs=self.n_cpus,
            memory_limit=self.memory_limit
        )

        print('Fitting')
        self.model.fit(X, y, dataset_name='GPP')
        print('Refitting')
        self.model.refit(X, y)
        print('Fitting done')
        
    def predict(self, X):
        return self.model.predict(X)
            