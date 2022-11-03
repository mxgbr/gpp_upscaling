#from random import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import sklearn
import sys
import os
import datetime as dt
import pickle
import glob
import copy
from collections import Counter, defaultdict
import random

PFT_REPLACEMENTS = pd.DataFrame({
    'MODIS': np.arange(1, 18),
    'New': ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'SH', 'SH', 'SAV', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CVM', 'SNO', 'BSV', 'WAT'],
    'Site': ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'CSH', 'OSH', 'WSA', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CVM', 'SNO', 'BSV', 'WAT'],   
})

def preprocess(df, var_set, cat=[], target=None, rm_all_nans=True):
    '''Performs standardized preprocessing tasks

    Selects variable sets, creates categorical dummy variables

    Args:
        df (pd.DataFrame): Data with variables as columns
        var_set (str): Indentifier of variable set, or list of column names
        cat (list): List of names of categorical variables
        target (str): Column name of target variable, or list of column names
        rm_all_nans (bool): Indicator if rows with all nans should be removed

    Returns:
        df_out: Data frame of selected variables
    '''
    if target is not None:
        target = df[target].copy()

    df_out = df.copy()

    if type(var_set) is list:
        df_out = df_out[setting]

    elif var_set == 'rs_min':
        # see https://daac.ornl.gov/VEGETATION/guides/FluxSat_GPP_FPAR.html
        df_out = df_out[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'BESS-PAR']]

    elif var_set == 'rs_min_vi':
        df_out = df_out[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'BESS-PAR', 'NDVI', 'EVI', 'GCI', 'NDWI', 'NIRv', 'kNDVI']]

    elif var_set == 'rs':
        df_out = df_out[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'LST_Day', 'LST_Night', 'Lai', 
                     'Fpar', 'CSIF-SIFdaily', 'CSIF-SIFinst', 'MODIS_LC', 'BESS-PAR', 'BESS-RSDN', 'BESS-PARdiff', 'ESACCI-sm', 'ET']]
        cat.append('MODIS_LC')

    elif var_set == 'rs_vi':
        df_out = df_out[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'NDVI', 'EVI', 'GCI', 'NDWI', 'NIRv', 'kNDVI', 'LST_Day', 'LST_Night', 'Lai', 
                     'Fpar', 'CSIF-SIFdaily', 'CSIF-SIFinst', 'MODIS_LC', 'BESS-PAR', 'BESS-RSDN', 'BESS-PARdiff', 'ESACCI-sm', 'ET']]
        cat.append('MODIS_LC')

    elif var_set == 'rs_meteo':
        df_out = df_out[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'LST_Day', 'LST_Night', 'Lai', 
                     'Fpar', 'CSIF-SIFdaily', 'CSIF-SIFinst', 'MODIS_LC', 'BESS-PAR', 'BESS-RSDN', 'BESS-PARdiff', 'ESACCI-sm', 'ET', 'total_precipitation', 'temperature_2m', 'vpd', 'prcp-lag3']]
        cat.append('MODIS_LC')

    elif setting == 'model0':
        # Yanghui
        df_out = df_out[['BESS-PAR','BESS-PARdiff','BESS-RSDN','PET','Ts','Tmean','prcp','vpd','prcp-lag3','ESACCI-sm','ET',
                     'b1','b2','b3','b4','b5','b6','b7','EVI','NDVI','GCI','NDWI','NIRv','kNDVI','CSIF-SIFdaily',
                     'Percent_Snow','Fpar','Lai','LST_Day','LST_Night', 'MODIS_LC']]

    elif setting == 'model1':
        # Yanghui
        df_out = df_out[['BESS-PAR','BESS-PARdiff','BESS-RSDN','PET','Ts','Tmean','prcp','vpd','prcp-lag3','ESACCI-sm','ET',
                     'b1','b2','b3','b4','b5','b6','b7','EVI','NDVI','GCI','NDWI','NIRv','kNDVI','CSIF-SIFdaily',
                     'Percent_Snow','Fpar','Lai','LST_Day','LST_Night', 'MODIS_LC']]

        df_out.loc[:, 'year'] = df_out.index.get_level_values('Date').year.values

    elif setting == 'model2':
        # Yanghui
        df_out = df_out[['BESS-PAR','BESS-PARdiff','BESS-RSDN','PET','Ts','Tmean','prcp','vpd','prcp-lag3','ESACCI-sm','ET',
                     'b1','b2','b3','b4','b5','b6','b7','EVI','NDVI','GCI','NDWI','NIRv','kNDVI','CSIF-SIFdaily',
                     'Percent_Snow','Fpar','Lai','LST_Day','LST_Night', 'MODIS_LC', 'CO2_concentration']]

        df_out.loc[:, 'year'] = df_out.index.get_level_values('Date').year.values

    # One-Hot Encoding
    cat_dfs = []
    for category in list(set(df_out.columns) & set(cat)):
        ohc = OneHotEncoder(sparse=False)
        X_cat = ohc.fit_transform(df_out[category].values.reshape(-1, 1))
        df_out = df_out.drop(category, axis=1)
        data_ohc = pd.DataFrame(np.array(X_cat), index=df_out.index, columns=[category + '_' + str(name) for name in ohc.categories_[0]])
        if np.nan in data_ohc.columns:
            data_ohc = data_ohc.drop(np.nan, axis=1)
        cat_dfs.append(data_ohc)

    df_out = pd.concat([df_out] + cat_dfs, axis=1)

    # remove all nans
    if rm_all_nans:
        df_out = df_out[df_out.notna().any(axis=1)]

    if target is not None:
        target = target.loc[df_out.index]

    return df_out, target

class Experiment(object):
    '''Organizes and logs model training and evaluation

    Experiments are identifiable by their ID, consisting of YYYYMMDD when the experiment was started

    Attributes:
        exp_id (str): Experiment ID
        path (str): Path to experiment folder
        output_dir (str): Location where experiments are stored
        logging (bool): Indicator if logging should be captured in a file
        prepend_date (bool): Indicator wether to prepend date to experiment ID

    TODO:
        exp index time-independent
    '''

    def __init__(self, exp_id='', output_dir='experiments/', logging=False, prepend_date=True, suffix=None):
        self.start = dt.datetime.now()
        self.suffix = suffix
        
        if exp_id == '':
            prepend_date = True
            exp_id = self.start.strftime("%H%M%S")
        
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''

        if prepend_date:
            prefix = self.start.strftime("%Y%m%d")
        else:
            prefix = ''

        exp_id = prefix + str(exp_id) + str(suffix)

        self.exp_id = exp_id
        self.path = os.path.join(output_dir, self.exp_id)
        print(self.path)

        # logging
        self.orig_stdout = sys.stdout
        self.stdout = None
        self.orig_stderr = sys.stderr

        if logging == True:
            self._create_folder()
            self.stdout = open(os.path.join(self.path, 'log.txt'), 'a')
            sys.stdout = self.stdout
            sys.stderr = self.stdout
            print('--------------------------------------------')
            print('Logging ', self.exp_id)

        print('Initialized:', self.start.strftime("%Y:%m:%d %H:%M:%S"))

    def _create_folder(self):
        '''Creates the experiment folder'''
        os.makedirs(self.path, exist_ok=True)

    def save(self, folds=1, X=None, y=None, params=None, models=None, train_idx=None, test_idx=None, y_pred=None, end_logging=True):
        '''Saves models and ouputs
        
        Saves the model results in the output folder
        
        Args:
            folds (int): Number of folds to be saved
            X (pd.DataFrame): Explanatory variables
            y (pd.Series): Target variables
            params (dict): Parameters
            models (list): List of trained models (instances of BaseModel)
            train_idx (list): List of pd.Series with training indices
            test_idx (list): List of pd.Series with test indices
            y_pred (list): List of pd.Series of predictions
            end_logging (bool): Indicator if logging should be reset to sys.stdout

        TODO:
            How save train_idx, test_idx?
        '''
        end = dt.datetime.now()
        print('Saved: ', end.strftime("%Y:%m:%d %H:%M:%S"))
        print('Duration: ', (end - self.start).total_seconds(), 's')

        # create experiment folder
        self._create_folder()

        # save training data
        if X is not None:
            X.to_csv(os.path.join(self.path, 'X.csv'))

        if y is not None:
            y.to_csv(os.path.join(self.path, 'y.csv'))

        # save parameter file
        if params is not None:
            with open(os.path.join(self.path, 'parameters.txt'), 'w') as f:
                print(params, file=f)

        # save each cv fold
        for idx in range(folds):

            # create fold folder
            dir = os.path.join(self.path, 'fold_' + str(idx))
            if not os.path.isdir(dir):
                os.mkdir(dir)

            # save indices
            if train_idx is not None:
                pd.Series(train_idx[idx], name='idx').to_csv(os.path.join(dir, 'train_idx.csv'), index=False)

            if test_idx is not None:
                pd.Series(test_idx[idx], name='idx').to_csv(os.path.join(dir, 'test_idx.csv'), index=False)

            # save predictions
            if y_pred is not None:
                y_pred[idx].to_csv(os.path.join(dir, 'y_pred.csv'))

            # save model
            if models is not None:
                models[idx].save(dir)

        if (end_logging == True) & (self.stdout is not None):
            sys.stdout = self.orig_stdout
            sys.stderr = self.orig_stderr
            self.stdout.close()

    @staticmethod
    def load(path):
        '''Loads models and ouptuts
        
        Returns:
            X (pd.DataFrame): Explanatory variables
            y (pd.Series): Target variables
            params (dict): Parameters
            models (list): List of trained models (instances of BaseModel)
            train_idx (list): List of lists with training indices
            test_idx (list): List of lists with test indices
            y_pred (list): List of pd.Series of predictions
        '''
        folds = len(glob.glob(os.path.join(path, 'fold_*')))

        X = None
        y = None
        params = None
        models = []
        train_idx = [] 
        test_idx = []
        y_pred = []
        
        # open X
        if os.path.isfile(os.path.join(path, 'X.csv')):
            X = pd.read_csv(os.path.join(path, 'X.csv'), index_col=[0, 1], parse_dates=True)

        # open y
        if os.path.isfile(os.path.join(path, 'y.csv')):
            y = pd.read_csv(os.path.join(path, 'y.csv'), index_col=[0, 1], parse_dates=True).squeeze()

        # open params
        if os.path.isfile(os.path.join(path, 'params.txt')):
            with open(os.path.join(path, 'params.txt'),'r') as inf:
                params = eval(inf.read())

        for idx in range(folds):
            # open models
            ## TODO

            # open train_idx
            if os.path.isfile(os.path.join(path, 'fold_' + str(idx), 'train_idx.csv')):
                train_idx.append(pd.read_csv(os.path.join(path, 'fold_' + str(idx), 'train_idx.csv')).squeeze().to_list())

            # open test_idx
            if os.path.isfile(os.path.join(path, 'fold_' + str(idx), 'test_idx.csv')):
                test_idx.append(pd.read_csv(os.path.join(path, 'fold_' + str(idx), 'test_idx.csv')).squeeze().to_list())

            # open y_pred
            if os.path.isfile(os.path.join(path, 'fold_' + str(idx), 'y_pred.csv')):
                y_pred.append(pd.read_csv(os.path.join(path, 'fold_' + str(idx), 'y_pred.csv'), index_col=[0, 1], parse_dates=True).squeeze())

        return X, y, params, models, train_idx, test_idx, y_pred

    def remove(self):
        '''Removes experiment'''
        pass

    def log(self, msg):
        '''Logs message
        
        Args:
            msg (str): Message
        '''
        pass