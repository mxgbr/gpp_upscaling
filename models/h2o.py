from models.basemodel import BaseModel
import h2o
from h2o.automl import H2OAutoML
import os
import re
import glob

class H2o(BaseModel):
    def __init__(self, 
                 time=60, 
                 port=54321, 
                 nthreads=-1, 
                 max_mem_size=None,
                 scoring='AUTO'
                ):
        
        super().__init__()
        self.time = time
        self.leaderboard = None
        self.port = port
        self.nthreads = nthreads
        self.max_mem_size = max_mem_size
        self.scoring = scoring

        h2o.init(port=port, nthreads=nthreads, max_mem_size=max_mem_size)

    def fit(self, X, y, oof_idx=None):
        train = X.copy()
        train['GPP'] = y
        train['groups'] = oof_idx
        self.model = None

        hf_train = h2o.H2OFrame(train)
        model = H2OAutoML(max_runtime_secs=self.time, sort_metric=self.scoring, stopping_metric=self.scoring)
        model.train(y='GPP', training_frame=hf_train, fold_column='groups')

        self.leaderboard = model.leaderboard.as_data_frame()
        self.model = h2o.get_model(model.leader.model_id)

    def predict(self, X):
        hf_test = h2o.H2OFrame(X)
        return self.model.predict(hf_test).as_data_frame()

    def save(self, path):
        # save model
        h2o.save_model(self.model, path=os.path.abspath(path), force=True)

        # save leaderboard
        self.leaderboard.to_csv(os.path.join(path, 'leaderboard.csv'))

    @staticmethod
    def load(path, init=True):
        path = glob.glob(os.path.join(path, '*_AutoML_*'))[0]
        path = os.path.abspath(path)

        if init:
            h2o.init(port=54321)
            
        return h2o.load_model(path)