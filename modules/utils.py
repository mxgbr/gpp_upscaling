#from random import random
import numpy as np
import pandas as pd
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from matplotlib import cm
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import datetime as dt
import pickle
import glob
import shutil
import copy
from collections import Counter, defaultdict
import random

from models.basemodel import BaseModel

PFT_REPLACEMENTS = pd.DataFrame({
    'MODIS': np.arange(1, 18),
    'New': ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'SH', 'SH', 'SAV', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CVM', 'SNO', 'BSV', 'WAT'],
    'Site': ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'CSH', 'OSH', 'WSA', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CVM', 'SNO', 'BSV', 'WAT'],   
})

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

    elif setting == 'joiner_7b+PAR':
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
    nan_mask_x = df_out.drop('GPP', axis=1).notna().any(axis=1)
    nan_mask_y = df_out['GPP'].notna()
    df_out = df_out[nan_mask_x & nan_mask_y]

    if strat is not None:
        strat = df[strat]
        strat = strat.loc[strat.index.intersection(df_out.index)]

    return df_out, strat

class Experiment(object):
    '''Organizes and logs model training and evaluation

    Experiments are identifiable by their ID, consisting of YYYYMMDDHHMMSS when the experiment was started

    Attributes:
        path (str): Path to experiment folder
        output_dir (str): Location where experiments are stored
        logging (bool): Indicator if logging should be captured in a file
    '''

    def __init__(self, exp_id=None, output_dir='experiments/', logging=False, suffix=None):
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''

        if exp_id is None:
            exp_id = dt.datetime.now().strftime("%Y%m%d%H%M%S") + suffix
        self.exp_id = exp_id
        self.path = os.path.join(output_dir, self.exp_id)
        print(self.path)

        # logging
        self.orig_stdout = sys.stdout
        self.stdout = None
        if logging == True:
            self._create_folder()
            self.stdout = open(os.path.join(path, 'log.txt'), 'a')
            sys.stdout = self.stdout
            print('--------------------------------------------')
            print('Logging ', self.exp_id)

    def _create_folder(self):
        '''Creates the experiment folder'''
        os.mkdir(self.path, exist_ok=True)

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
                train_idx[idx].to_csv(os.path.join(dir, 'train_idx'))

            if test_idx is not None:
                test_idx[idx].to_csv(os.path.join(dir, 'test_idx'))

            # save predictions
            if pred is not None:
                pred[idx].to_csv(os.path.join(dir, 'y_pred'))

            # save model
            if models is not None:
                models[idx].save(dir)

    def load(self):
        '''Loads models and ouptuts
        
        Returns:
            X (pd.DataFrame): Explanatory variables
            y (pd.Series): Target variables
            params (dict): Parameters
            models (list): List of trained models (instances of BaseModel)
            train_idx (list): List of pd.Series with training indices
            test_idx (list): List of pd.Series with test indices
            y_pred (list): List of pd.Series of predictions
        '''
        folds = glob.glob(os.path.join(self.path, 'fold_*'))

        X = y = params = None
        models = train_idx = test_idx = y_pred = []
        
        # open X
        if os.path.isfile(os.path.join(self.path, 'X.csv')):
            X = pd.read_csv(os.path.join(self.path, 'X.csv'), index_col=[0, 1], parse_dates=True)

        # open y
        if os.path.isfile(os.path.join(self.path, 'y.csv')):
            y = pd.read_csv(os.path.join(self.path, 'y.csv'), index_col=[0, 1], parse_dates=True).squeeze()

        # open params
        if os.path.isfile(os.path.join(self.path, 'params.txt')):
            with open(os.path.join(self.path, 'params.txt'),'r') as inf:
                params = eval(inf.read())

        for idx in folds:
            # open models
            ## TODO

            # open train_idx
            if os.path.isfile(os.path.join(self.path, 'fold_' + idx, 'train_idx.csv')):
                train_idx.append(pd.read_csv(os.path.join(self.path, 'fold_' + idx, 'train_idx.csv')))

            # open test_idx
            if os.path.isfile(os.path.join(self.path, 'fold_' + idx, 'test_idx.csv')):
                test_idx.append(pd.read_csv(os.path.join(self.path, 'fold_' + idx, 'test_idx.csv')))

            # open y_pred
            if os.path.isfile(os.path.join(self.path, 'fold_' + idx, 'y_pred.csv')):
                y_pred.append(pd.read_csv(os.path.join(self.path, 'fold_' + idx, 'y_pred.csv'), index_col=[0, 1], parse_dates=True).squeeze())

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

def tune(func, df, y_col, strat, n_folds=5, n_folds_inner=5, num_cpus=None, use_ray=True, random_state=2, **kwargs):
    '''
    TODO:
        Not needed anymore
    '''
    # define groups for outer and inner sampling
    groups = df.index.get_level_values(0).values
    sgkf_outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state) 

    # predictors
    X = df.drop(y_col, axis=1)
    y = df[y_col]

    def ray_func(train_idx, test_idx, X, y, strat, groups, idx):
        '''Performs outer CV loop'''

        # train and test sets
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # corresponding group and strat column
        strat_inner = strat.iloc[train_idx]
        groups_inner = groups[train_idx]

        oof_idx = np.full(y_train.shape, -1)
        sgkf_inner = StratifiedGroupKFold(n_splits=n_folds_inner, shuffle=True, random_state=random_state)
        for idx_inner, (train_inner, test_inner) in enumerate(sgkf_inner.split(X_train, strat_inner, groups=groups_inner)):
            oof_idx[test_inner] = idx_inner

        # model prediction function
        model, y_pred = func(X_train, y_train, X_test, oof_idx, idx, **kwargs)

        # evaluation
        metrics = [sklearn.metrics.r2_score(y_test, y_pred), sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)]

        # save
        return_dict = {
            'id': idx,
            'model': model,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'pred': y_pred,
            'metrics': metrics
        }
        return return_dict

    if use_ray == True:
        ray.shutdown()
        ray.init(num_cpus=num_cpus)
        ray_func = ray.remote(ray_func)

    futures = []
    for idx, (train_idx, test_idx) in enumerate(sgkf_outer.split(X, strat, groups=groups)): 
        if use_ray:
            futures.append(ray_func.remote(train_idx, test_idx, X, y, strat, groups, idx))
        else:
            futures.append(ray_func(train_idx, test_idx, X, y, strat, groups, idx))
            
    if use_ray:
        runs = ray.get(futures)
    else:
        runs = futures

    return runs, X, y

def split_sensitivity(tune_func, data_sel, y_var, strat, num_cpus=None, n_folds=5, random_states=20, params=None):
    '''
    TODO:
        Not needed anymore
    '''
    split_states = np.random.randint(0, 1000, size=random_states)

    metrics = []
    for idx, state in enumerate(split_states):
        print('Loop ' + str(idx))
        results, X, y = tune(tune_func, data_sel, y_var, strat, num_cpus=params['num_cpus'], n_folds=5, random_state=state, params=params)

        y_eval = []
        for ii in results:
            test_idx = ii['test_idx']
            y_test = y.iloc[test_idx]
            y_pred = ii['pred']
            y_eval.append(pd.DataFrame({'Pred': y_pred.flatten(), 'GT': y_test}, index=y.iloc[test_idx].index))

        y_eval = pd.concat(y_eval)
        metrics.append(sklearn.metrics.r2_score(y_eval.GT.values, y_eval.Pred.values))
        
    return np.mean(metrics), np.std(metrics)


def split_sensitivity_single(tune_func, data_sel, y_var, strat, num_cpus=None, n_folds=5, random_state=None, params=None, min_months=24, idx=None, exp_name='', cols={}):
    '''Runs split sensitivity test, single process
    
    Attributes:
        tune_func: Function containing model fitting and prediction logic.
    
    TODO:
        Not needed anymore
    '''

    if random_state is None:
        random_state = np.random.randint(0, 1000)
    
    # run model
    results, X, y = tune(tune_func, data_sel, y_var, strat, num_cpus=num_cpus, n_folds=n_folds, random_state=random_state, params=params)

    # calc metrics r2 and rmse
    y_eval = []
    for ii in results:
        test_idx = ii['test_idx']
        y_test = y.iloc[test_idx]
        y_pred = ii['pred']
        y_eval.append(pd.DataFrame({'Pred': y_pred.flatten(), 'GT': y_test}, index=y.iloc[test_idx].index))

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

def annual_mean(ts, transform=False):
    '''Calculates annual mean values'''
    grp = ts.groupby(['SITE_ID', ts.index.get_level_values('Date').year])
    if transform:
        return grp.transform('mean')
    return grp.mean()

def across_site_variability(ts):
    return ts.groupby('SITE_ID').mean()

def iav(ts, detrend=False):
    ts = ts.sub(msc(ts, transform=True))
    if detrend == True:
        ts = ts.sub(trend(ts))

    return ts

def msc(ts, transform=False, no_mean=False):
    if no_mean:
        ts = ts.sub(ts.groupby(['SITE_ID']).transform('mean'))
    grp = ts.groupby(['SITE_ID', ts.index.get_level_values('Date').month])
    if transform:
        return grp.transform('mean')
    return grp.mean()

def lr_model(series_inp, return_coef=False):
    series = series_inp.droplevel(0)
    x = ((series.index - pd.to_datetime(date(1970, 1, 31))) / np.timedelta64(1, 'M')).values.round().reshape(-1, 1)
    y = series.values
    lr = LinearRegression()
    lr.fit(x, y)
    if return_coef == True:
            return lr.coef_
    return pd.Series(lr.predict(x), index=series_inp.index)

def trend(ts):
    '''Calculates trend and intercept
    
    Groups by sites in performs a linear regression for each site

    Args:
        ts (DataFrame): time series with sites and dates as index

    Returns:
        Slope in pd.Series
    '''     
    grp = ts.groupby(['SITE_ID']).apply(lr_model)
    return grp

def across_site_trend(ts):
    '''
    Calculates trend on site level from linear regression

    Args:
        ts (DataFrame): time series with sites and dates as index

    Returns:
        pd.Series with trend-only values
    '''
    grp = ts.groupby('SITE_ID').apply(lr_model, return_coef=True)
    return pd.Series(np.concatenate(grp.values).ravel(), index=grp.index)

def evaluation_plot(y_eval):
    '''
    Creates evaluation plots for model performance

    Args:
        y_eval (pd.DataFrame): DataFrame with ground truth values in `GT` column and predictions in `Pred` column. Index must consist of SITE_ID and Date (datetimeindex)
    '''
    r2_overall = sklearn.metrics.r2_score(y_eval.GT.values, y_eval.Pred.values)
    r2_trend = sklearn.metrics.r2_score(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values)
    # r2_seasonal = sklearn.metrics.r2_score(seasonal_cycle_mean(y_eval.GT).values, seasonal_cycle_mean(y_eval.Pred).values)
    r2_anomalies = sklearn.metrics.r2_score(iav(y_eval.GT, detrend=True).values, iav(y_eval.Pred, detrend=True).values)
    r2_sites = sklearn.metrics.r2_score(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values)
    r2_iav = sklearn.metrics.r2_score(iav(y_eval.GT).values, iav(y_eval.Pred).values)
    r2_msc = sklearn.metrics.r2_score(msc(y_eval.GT).values, msc(y_eval.Pred).values)

    rmse_overall = sklearn.metrics.mean_squared_error(y_eval.GT.values, y_eval.Pred.values, squared=False)
    rmse_trend = sklearn.metrics.mean_squared_error(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values, squared=False)
    # rmse_seasonal = sklearn.metrics.mean_squared_error(seasonal_cycle_mean(y_eval.GT).values, seasonal_cycle_mean(y_eval.Pred).values, squared=False)
    rmse_anomalies = sklearn.metrics.mean_squared_error(iav(y_eval.GT, detrend=True).values, iav(y_eval.Pred).values, squared=False)
    rmse_sites = sklearn.metrics.mean_squared_error(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values, squared=False)
    rmse_iav = sklearn.metrics.mean_squared_error(iav(y_eval.GT).values, iav(y_eval.Pred).values, squared=False)
    rmse_msc = sklearn.metrics.mean_squared_error(msc(y_eval.GT).values, msc(y_eval.Pred).values, squared=False)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # plot 1
    x = y_eval.Pred
    y = y_eval.GT
    r2_plot(x, y, ax[0, 0], r2=r2_overall, rmse=rmse_overall)
    ax[0, 0].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[0, 0].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[0, 0].set_title('Overall prediction')

    # plot 2
    x = across_site_trend(y_eval.Pred)
    y = across_site_trend(y_eval.GT)
    r2_plot(x, y, ax[0, 1], r2=r2_trend, rmse=rmse_trend)
    ax[0, 1].set_xlabel('Predicted slope GPP [$gC m^{-2} d^{-1} month^{-1}$]')
    ax[0, 1].set_ylabel('FLUXNET slope GPP [$gC m^{-2} d^{-1} month^{-1}$]')
    ax[0, 1].set_title('Trend')

    # plot 3
    # x = seasonal_cycle_mean(y_eval.Pred)
    # y = seasonal_cycle_mean(y_eval.GT)
    # r2_plot(x, y, ax[0, 1], r2=r2_seasonal, rmse=rmse_seasonal)
    # ax[0, 1].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    # ax[0, 1].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    # ax[0, 1].set_title('Seasonal cycle')

    # plot 5
    x = across_site_variability(y_eval.Pred)
    y = across_site_variability(y_eval.GT)
    r2_plot(x, y, ax[0, 2], r2=r2_sites, rmse=rmse_sites)
    ax[0, 2].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[0, 2].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[0, 2].set_title('Across-site variability')

    # plot 7
    x = msc(y_eval.Pred)
    y = msc(y_eval.GT)
    im = r2_plot(x, y, ax[1, 0], r2=r2_msc, rmse=rmse_msc)
    ax[1, 0].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 0].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 0].set_title('Mean seasonal cycle')

    # plot 6
    x = iav(y_eval.Pred)
    y = iav(y_eval.GT)
    r2_plot(x, y, ax[1, 1], r2=r2_iav, rmse=rmse_iav)
    ax[1, 1].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 1].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 1].set_title('Interanual variability')

    # plot 4
    x = iav(y_eval.Pred, detrend=True)
    y = iav(y_eval.GT, detrend=True)
    r2_plot(x, y, ax[1, 2], r2=r2_anomalies, rmse=rmse_anomalies)
    ax[1, 2].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 2].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 2].set_title('Interannual variability (detrended)')

    # ax_cbar = fig.add_axes([0.3, 0.1, 0.4, 0.03])
    # plt.colorbar(im, cax=ax_cbar, orientation='horizontal')

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    return fig
