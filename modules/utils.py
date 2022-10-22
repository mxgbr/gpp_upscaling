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

    elif var_set == 'rs':
        df_out = df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'NDVI', 'EVI', 'GCI', 'NDWI', 'NIRv', 'kNDVI', 'LST_Day', 'LST_Night', 'Lai', 
                     'Fpar', 'CSIF-SIFdaily', 'CSIF-SIFinst', 'MODIS_LC', 'BESS-PAR', 'BESS-RSDN', 'BESS-PARdiff', 'ESACCI-sm']]
        cat.append('MODIS_LC')

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

def select_vars(df, setting, gpp=None, strat=None):
    '''selects different predictors for different experiment set-ups'''
    
    cat = []

    if type(setting) is list:
        df_out = df[setting]
        df_out.loc[:, 'year'] = df_out.index.get_level_values('Date').year.values

    elif setting == 'tramontana_RS':
        df_out = df[['MODIS_LC', 'NDWI', 'LST_Day', 'LST_Night']].copy()
        df_out['EVI_amp_msc'] = msc(df['EVI'], transform=True, no_mean=True).max()
        df_out['b7_amp_msc'] = msc(df['b7'], transform=True, no_mean=True).max()
        df_out['LST_Day_max_msc'] = msc(df['LST_Day'], transform=True).max()
        df_out['LAI_msc'] = msc(df['LAI'], transform=True).max()
        # NDVI x incoming radiation (Rpot)

        cat.append('MODIS_LC')

    elif setting == 'tramontana_RS+METEO':
        df_out = df[['MODIS_LC', 'Tair']]
        df_out['NDVI_amp_msc'] = msc(df['NDVI'], transform=True, no_mean=True).max()
        df_out['b4_amp_msc'] = msc(df['b4'], transform=True, no_mean=True).max()
        df_out['NDWI_amp_msc'] = msc(df['NDWI'], transform=True, no_mean=True).max()
        # water availability lower amplitude of MSC
        # WAI_L
        df_out['LST_Night_msc'] = msc(df['LST_Night'], transform=True).max()
        df_out['(Fpar,LST_Day)_msc'] = msc(df['Fpar'] * df['LST_Day'], transform=True).max()
        # EVI x Rpot MSC
        # incoming radiation (Rpot) x (MSC of NDVI)

        cat.append('MODIS_LC')

    elif setting == 'me_RS':
        df_out = df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'NDVI', 'EVI', 'GCI', 'NDWI', 'NIRv', 'kNDVI', 'LST_Day', 'LST_Night', 'Lai', 
                     'Fpar', 'CSIF-SIFdaily', 'CSIF-SIFinst', 'MODIS_LC', 'BESS-PAR', 'BESS-RSDN', 'BESS-PARdiff']]
                     # ESACCI-sm
                     # 'evaporation_from_vegetation_transpiration'
        df_out.loc[:, 'year'] = df_out.index.get_level_values('Date').year.values

        cat.append('MODIS_LC')

    elif setting == 'me_RS+METEO':
        df_out = df[['total_precipitation', 'temperature_2m', 'vpd', 'prcp-lag3', 'b1', 'b2', 'b3', 'b4', 'b5', 
                'b6', 'b7', 'NDVI', 'EVI', 'GCI', 'NDWI', 'NIRv', 'kNDVI', 'LST_Day', 'LST_Night', 'Lai', 
                'Fpar', 'evaporation_from_vegetation_transpiration', 'ESACCI-sm', 'BESS-PAR', 'BESS-RSDN', 
                'BESS-PARdiff', 'CSIF-SIFdaily', 'CSIF-SIFinst', 'MODIS_LC']]
        df_out.loc[:, 'year'] = df_out.index.get_level_values('Date').year.values

        cat.append('MODIS_LC')

    elif setting == 'rs_min':
        # see https://daac.ornl.gov/VEGETATION/guides/FluxSat_GPP_FPAR.html
        df_out = df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'BESS-PAR']]

    elif setting == 'model0':
        # Yanghui
        df_out = df[['BESS-PAR','BESS-PARdiff','BESS-RSDN','PET','Ts','Tmean','prcp','vpd','prcp-lag3','ESACCI-sm','ET',
                     'b1','b2','b3','b4','b5','b6','b7','EVI','NDVI','GCI','NDWI','NIRv','kNDVI','CSIF-SIFdaily',
                     'Percent_Snow','Fpar','Lai','LST_Day','LST_Night', 'MODIS_LC']]

    elif setting == 'model1':
        # Yanghui
        df_out = df[['BESS-PAR','BESS-PARdiff','BESS-RSDN','PET','Ts','Tmean','prcp','vpd','prcp-lag3','ESACCI-sm','ET',
                     'b1','b2','b3','b4','b5','b6','b7','EVI','NDVI','GCI','NDWI','NIRv','kNDVI','CSIF-SIFdaily',
                     'Percent_Snow','Fpar','Lai','LST_Day','LST_Night', 'MODIS_LC']]

        df_out.loc[:, 'year'] = df_out.index.get_level_values('Date').year.values

    elif setting == 'model2':
        # Yanghui
        df_out = df[['BESS-PAR','BESS-PARdiff','BESS-RSDN','PET','Ts','Tmean','prcp','vpd','prcp-lag3','ESACCI-sm','ET',
                     'b1','b2','b3','b4','b5','b6','b7','EVI','NDVI','GCI','NDWI','NIRv','kNDVI','CSIF-SIFdaily',
                     'Percent_Snow','Fpar','Lai','LST_Day','LST_Night', 'MODIS_LC', 'CO2_concentration']]

        df_out.loc[:, 'year'] = df_out.index.get_level_values('Date').year.values

    if gpp is not None:
        df_out = df_out.merge(gpp, left_index=True, right_index=True)
    else:
        df_out.loc[:, 'GPP'] = df['GPP_NT_VUT_REF']

    # One-Hot Encoding
    for category in cat:
        ohc = OneHotEncoder(sparse=False)
        X_cat = ohc.fit_transform(df_out[category].values.reshape(-1, 1))
        df_out = df_out.drop(category, axis=1)
        data_ohc = pd.DataFrame(np.array(X_cat), index=df_out.index, columns=[category + '_' + str(name) for name in ohc.categories_[0]])
        if np.nan in data_ohc.columns:
            data_ohc = data_ohc.drop(np.nan, axis=1)
        df_out = pd.concat([df_out, data_ohc], axis=1)

    # drop nans (where all x nan or y nan)
    ## TODO should not be necessary
    nan_mask_x = df_out.drop('GPP', axis=1).notna().any(axis=1)
    nan_mask_y = df_out['GPP'].notna()
    df_out = df_out[nan_mask_x & nan_mask_y]

    if strat is not None:
        strat = df[strat]
        strat = strat.loc[strat.index.intersection(df_out.index)]

    return df_out, strat

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

        exp_id = prefix + str(exp_id) + suffix

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

    def save(self, folds, X=None, y=None, params=None, models=None, train_idx=None, test_idx=None, y_pred=None, end_logging=True):
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

def eval_sensitivity(runs, y, random_state, min_months=24, cols={}, exp_name='', idx=None):
    # calc metrics r2 and rmse
    y_eval = []
    for ii in runs:
        test_idx = ii['test_idx']
        y_test = y.iloc[test_idx].values.flatten()
        y_pred = ii['pred']
        y_eval.append(pd.DataFrame({'Pred': y_pred, 'GT': y_test}, index=y.iloc[test_idx].index))

    y_eval = pd.concat(y_eval)

    # filter min months
    y_eval = y_eval[(y_eval.GT.groupby('SITE_ID').transform(lambda x: x.groupby('SITE_ID').transform(lambda x: x.count())) >= min_months)]

    out_df = [
        sklearn.metrics.r2_score(y_eval.GT.values, y_eval.Pred.values),
        sklearn.metrics.r2_score(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values),
        sklearn.metrics.r2_score(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values),
        sklearn.metrics.r2_score(msc(y_eval.GT).values, msc(y_eval.Pred).values),
        sklearn.metrics.r2_score(iav(y_eval.GT).values, iav(y_eval.Pred).values),
        sklearn.metrics.mean_squared_error(y_eval.GT.values, y_eval.Pred.values, squared=False),
        sklearn.metrics.mean_squared_error(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values, squared=False),
        sklearn.metrics.mean_squared_error(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values, squared=False),
        sklearn.metrics.mean_squared_error(msc(y_eval.GT).values, msc(y_eval.Pred).values, squared=False),
        sklearn.metrics.mean_squared_error(iav(y_eval.GT).values, iav(y_eval.Pred).values, squared=False),
        random_state
    ]

    out_df = pd.DataFrame([out_df + list(cols.values())], index=[idx], columns=['r2_overall', 'r2_trend', 'r2_sites', 'r2_msc', 'r2_iav', 'rmse_overall', 'rmse_trend', 'rmse_sites', 'rmse_msc', 'rmse_iav', 'random_state'] + list(cols.keys()))
    
    # create metrics csv if non-existent
    out_dir = 'output_multi'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # append metrics to metrics csv
    out_path = os.path.join(out_dir, 'metrics_' + exp_name + '.csv')
    out_df.to_csv(out_path, mode='a', header=not os.path.exists(out_path))

def r2_plot(x, y, ax, r2=None, rmse=None, n_bins=40, line=True, threshold=0.005):
    cmap = cm.get_cmap('viridis').copy()
    cmap.set_under('w')
    vmin = np.minimum(y.min(), x.min())
    vmax = np.maximum(y.max(), x.max())
    if line == True:
        ax.plot([vmin, vmax], [vmin, vmax], c='k')
    hist, xbins, ybins, im = ax.hist2d(x, y, bins=np.linspace(vmin, vmax, n_bins), cmap=cmap, density=True, vmin=threshold)
    x_bin_width = xbins[1] - xbins[0]
    y_bin_width = ybins[1] - ybins[0]
    x_bins = ((hist < threshold) & (hist > 0)).nonzero()[0] * x_bin_width + x_bin_width/2 + vmin
    y_bins = ((hist < threshold) & (hist > 0)).nonzero()[1] * y_bin_width + y_bin_width/2 + vmin
    # print((hist == 1).nonzero())
    # print(x_bins, y_bins)
    # print(hist)
    ax.scatter(x_bins, y_bins, s=3, c='gray', marker='o')
    #ax.axis('equal')
    ax.set(adjustable='box', aspect='equal')

    loc = 0.95
    if rmse is not None:
        ax.text(0.05, loc, r'$RMSE=%.2f$' % (rmse, ), transform=ax.transAxes, fontsize=14, verticalalignment='top')
        loc -= 0.08

    if r2 is not None:
        ax.text(0.05, loc, r'$r^2=%.2f$' % (r2, ), transform=ax.transAxes, fontsize=14, verticalalignment='top')

    return im

## filter site-years with not enough data