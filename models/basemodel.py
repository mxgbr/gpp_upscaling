import pickle
import os

class BaseModel(object):
    '''The BaseModel object contains the basic structure for models in the CV
    
    Attributes:
        model: Instance of the trained model
        id (int): Unique identifier

    TODO:
        save and load functions?
    '''
    def __init__(self):
        self.model = None
        self.id = None
        self.from_file = False

    def preproc(self, df, y_col, strat):
        '''Placeholder for model-specific pre-processing
        
        Can be used, e.g., if NaNs need to be filled. This function is a dummy function and can be overwritten if actual pre-processing
        is applied.
        
        Args:
            df (pd.DataFrame): DataFrame with variables in columns, including the target variable
            y_col (str): Name of the target variable
            strat (pd.Series): Series of classes for stratification
            
        Returns:
            df: DataFrame after pre-processing
            strat: Series of classes for stratification
            
        TODO:
            Clarify role of strat, why separate?
        '''
        return df, strat

    def fit(self, X, y, oof_idx=None):
        '''Fits the model
        
        Args:
            X (pd.DataFrame): Dataframe with predictor variables
            y (pd.Series): Series with target variable
            oof_idx (pd.Series): Index groups, can be used by the model for cv tuning. Enables stratified and grouped train/val splits, e.g., with PredefinedSplit.
        '''
        pass

    def predict(self, X):
        '''Predicts from the fitted model
        
        Args:
            X (pd.DataFrame): Frame in same structure as for fitting
        '''
        pass

    def save(self, path):
        '''Saves the model

        Pickles by default, should be overwritten if custom saving methods applied

        Args:
            path (str): Saving directory
        '''
        with open(os.path.join(path, 'model'), 'ab') as outfile:
            pickle.dump(self.model, outfile)

    @classmethod
    def load(cls, path, **params):
        '''Loads a model

        Model loaded is of type BaseModel with default attributes and the loaded model in the model attribute

        Args:
            path (str): Loading directory
            **params: Parameters for model initialization

        Returns:
            BaseModel object
        '''
        model = pickle.load(open(path, 'model'), 'rb')

        model_wrapper = cls(**params)
        model_wrapper.model = model
        model_wrapper.from_file = True

        return model_wrapper