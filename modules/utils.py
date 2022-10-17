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

class StratifiedGroupKFold():
    '''
    Manual implementation, as not included in sklearn 0.24

    See https://www.kaggle.com/code/jakubwasikowski/stratified-group-k-fold-cross-validation/notebook
    '''
    def __init__(self, n_splits=5, shuffle=True, random_state=2):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        lenc = LabelEncoder()
        y = lenc.fit_transform(y)
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
        
        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

class Experiment(object):
    def __init__(self, path=None, output_dir='output/', logging=False, desc='', suffix=''):
        '''Initializes experiment
        
        If path is empty, a new dir is created for the experiment with current time stamp
        '''
        if path is None:
            self.time_id = dt.datetime.now().strftime("%d%m%Y-%H%M%S") + suffix
            path = os.path.join(output_dir, self.time_id)
            os.mkdir(path)
            print('Created experiment directory ' + path)
            self._append_summary_file(os.path.join(output_dir, 'experiments.csv'), [desc, 'STARTED'], ['Description', 'Status'])
        else:
            self.time_id = os.path.basename(os.path.normpath(path))
            print('Working in existing directory ' + path)

        self.path = path
        ## TODO desc empty of working in existing directory
        self.desc = desc
        self.output_dir = output_dir
        self.time_id = self.time_id

        self.orig_stdout = sys.stdout
        self.stdout = None
        if logging == True:
            self.stdout = open(os.path.join(path, 'log.txt'), 'a')
            sys.stdout = self.stdout
            print('--------------------------------------------')
            print('Logging ', self.time_id)

    def save(self, y, X=None, runs=[], params=None, end_logging=True, save_func=None):
        '''Saves models, parameters, and metrics'''

        # save predictors (if available)
        if X is not None:
            X.to_csv(os.path.join(self.path, 'X.csv'))

        # save GPP measurements
        y.to_csv(os.path.join(self.path, 'y.csv'))
        
        # save parameters if available
        if params is not None:
            with open(os.path.join(self.path, 'parameters.txt'), 'w') as f:
                print(params, file=f)

        if (end_logging == True) & (self.stdout is not None):
            sys.stdout = self.orig_stdout
            self.stdout.close()

        # save runs
        for idx, run in enumerate(runs):
            dir = os.path.join(self.path, 'run_' + str(idx))
            
            if not os.path.isdir(dir):
                os.mkdir(dir)

            # save training indices for X and y
            if 'train_idx' in run:
                np.save(os.path.join(dir, 'train_idx'), run['train_idx'])

            # save test indices for X and y; if not available use all
            if 'test_idx' in run:
                np.save(os.path.join(dir, 'test_idx'), run['test_idx'])
            else:
                np.save(os.path.join(dir, 'test_idx'), np.arange(0, len(y)))

            # save model prediction (corresponding to test_idx)
            if 'pred' in run:
                np.save(os.path.join(dir, 'pred'), run['pred'])

            # save model as pickle or with custom model save function (with parameters model, path)
            if 'model' in run:
                if run['model'] is not None:
                    if save_func is None:
                        with open(os.path.join(dir, 'model'), 'ab') as outfile:
                            pickle.dump(run['model'], outfile)
                    else:
                        save_func(run['model'], dir)

        if self.desc == '':
            desc = np.nan
        else:
            desc = self.desc
        self._append_summary_file(os.path.join(self.output_dir, 'experiments.csv'), [desc, 'COMPLETE'], ['Description', 'Status'])

        # output_path=os.path.join(self.output_dir, 'experiments.csv')
        # experiments = pd.DataFrame({'id': [os.path.basename(os.path.normpath(self.path))], 'desc': [self.desc], 'saved': [dt.datetime.now().strftime("%d%m%Y-%H%M%S")]})
        # experiments.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

        # print('Saved to ', self.path)

    def load(self, concat=True):
        '''Loads all predictions and returns data frame with GT and prediction'''
        y = pd.read_csv(os.path.join(self.path, 'y.csv'), index_col=[0, 1], parse_dates=True).squeeze()

        runs = glob.glob(os.path.join(self.path, 'run_*'))

        y_eval = []
        for ii in runs:
            test_idx = np.load(os.path.join(ii, 'test_idx.npy'))
            y_test = y.iloc[test_idx]
            y_pred = np.load(os.path.join(ii, 'pred.npy'))
            y_eval.append(pd.DataFrame({'Pred': y_pred.flatten(), 'GT': y_test}, index=y.iloc[test_idx].index))

        if concat == True:
            y_eval = pd.concat(y_eval)

        return y_eval

    def remove(self):
        try:
            shutil.rmtree(self.path)

            # rm from experiments
            path = os.path.join(self.output_dir, 'experiments.csv')
            exp = pd.read_csv(path, index_col=0, parse_dates=True).drop(self.time_id, errors='ignore')
            exp.to_csv(path)

            # rm from evaluation
            path = os.path.join(self.output_dir, 'evaluation.csv')
            exp = pd.read_csv(path, index_col=0, parse_dates=True).drop(self.time_id, errors='ignore')
            exp.to_csv(path)

        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    def evaluate(self, filter=None, min_months=2):
        '''Creates evaluation charts and statistics'''
        
        y_eval = self.load()

        # filter min months
        y_eval = y_eval[(y_eval.GT.groupby('SITE_ID').transform(lambda x: x.groupby('SITE_ID').transform(lambda x: x.count())) >= min_months)]

        path = os.path.join(self.path, 'evaluation')
        if not os.path.isdir(path):
            os.mkdir(path)

        # create temporal analysis
        fig = evaluation_plot(y_eval)
        fig.savefig(os.path.join(path, 'evaluation_temp.jpg'))

        # create analysis on Biome/Climate level

        # Log analysis
        r2 = sklearn.metrics.r2_score(y_eval.GT.values, y_eval.Pred.values)
        rmse = sklearn.metrics.mean_squared_error(y_eval.GT.values, y_eval.Pred.values, squared=False)

        self._append_summary_file(os.path.join(self.output_dir, 'evaluation.csv'), [self.desc, r2, rmse], ['Description', 'R2', 'RMSE'])

        # evaluation = pd.DataFrame({'ID': [self.time_id], 'Description': [self.desc], 'R2': [r2], 'RMSE': [rmse], 'Date': [dt.datetime.now().strftime("%d%m%Y-%H%M%S")]}).set_index('ID')
        # if os.path.isfile(os.path.join(self.output_dir, 'evaluation.csv')):
        #     evaluation = evaluation.combine_first(pd.read_csv(os.path.join(self.output_dir, 'evaluation.csv'), index_col=0, parse_dates=True))
        
        # evaluation.to_csv(os.path.join(self.output_dir, 'evaluation.csv'))
    
    def evaluate_each(self):
        y_eval = self.load(concat=False)

        r2 = []
        rmse = []
        for ii in y_eval:
            r2.append(sklearn.metrics.r2_score(ii.GT.values, ii.Pred.values))
            rmse.append(sklearn.metrics.mean_squared_error(ii.GT.values, ii.Pred.values, squared=False))

        return r2, rmse

    def evaluate_stats(self, min_months=24):

        y_eval = self.load()

        # filter min months
        y_eval = y_eval[(y_eval.GT.groupby('SITE_ID').transform(lambda x: x.groupby('SITE_ID').transform(lambda x: x.count())) >= min_months)]

        return {
            'r2_overall': sklearn.metrics.r2_score(y_eval.GT.values, y_eval.Pred.values),
            'r2_trend': sklearn.metrics.r2_score(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values),
            'r2_sites': sklearn.metrics.r2_score(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values),
            'r2_msc': sklearn.metrics.r2_score(msc(y_eval.GT).values, msc(y_eval.Pred).values),
            'r2_iav': sklearn.metrics.r2_score(iav(y_eval.GT).values, iav(y_eval.Pred).values)
        }

    def log(self, msg):
        '''Logs message to log file'''
        with open(os.path.join(self.path, 'log.txt'), 'a') as f:
            f.write(msg)

    def _append_summary_file(self, path, data, columns):
        '''data can use nan to keep previous records'''
        # TODO does appending work correctly?

        data = [self.time_id] + data + [dt.datetime.now().strftime("%d.%m.%Y %H:%M:%S")]
        columns = ['ID'] + columns + ['Save Date']

        evaluation = pd.DataFrame([data], columns=columns).set_index('ID')
        if os.path.isfile(path):
            evaluation = evaluation.combine_first(pd.read_csv(path, index_col=0, parse_dates=True))
        
        evaluation.to_csv(path)

class CV(object):
    '''Runs CV on model'''
    def __init__(self, model, n_folds_cv=5, n_folds_tuning=5, random_state=2, use_ray=True, num_cpus=None):
        self.model = model
        self.n_folds_cv = n_folds_cv
        self.n_folds_tuning = n_folds_tuning
        self.random_state = random_state
        self.use_ray = use_ray
        self.num_cpus = num_cpus

    def fit_predict_single(self, X, y, train_idx, test_idx, groups, strat=None):
        '''Fits, predicts and evaluates one model'''
        # train and test sets
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # corresponding group and strat column
        strat_inner = strat.iloc[train_idx]
        groups_inner = groups[train_idx]

        # CV indices for internal hyperparameter tuning by the model
        oof_idx = np.full(y_train.shape, -1)
        sgkf_inner = StratifiedGroupKFold(n_splits=self.n_folds_tuning, shuffle=True, random_state=self.random_state)
        for idx_inner, (train_inner, test_inner) in enumerate(sgkf_inner.split(X_train, strat_inner, groups=groups_inner)):
            oof_idx[test_inner] = idx_inner

        # model prediction function
        predictor = copy.deepcopy(self.model)
        predictor.fit(X_train, y_train, oof_idx)
        y_pred = predictor.predict(X_test)
        y_pred = np.array(y_pred).flatten()

        # evaluation
        metrics = [sklearn.metrics.r2_score(y_test, y_pred), sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)]

        # save
        return_dict = {
            #'id': idx,
            'model': predictor,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'pred': y_pred,
            'metrics': metrics
        }
        return return_dict
    
    def fit_predict(self, df, y_col, strat=None):
        '''Runs CV loop n_folds_cv times'''

        df, strat = self.model.preproc(df, y_col, strat)

        # define groups for outer and inner sampling
        groups = df.index.get_level_values(0).values
        sgkf_outer = StratifiedGroupKFold(n_splits=self.n_folds_cv, shuffle=True, random_state=self.random_state) 

        # predictors
        X = df.drop(y_col, axis=1)
        y = df[y_col]

        if self.use_ray == True:
            ray.shutdown()
            ray.init(num_cpus=self.num_cpus)
            ray_func = ray.remote(fit_predict_single_ray)

        futures = []
        for idx, (train_idx, test_idx) in enumerate(sgkf_outer.split(X, strat, groups=groups)): 
            if self.use_ray:
                ## TODO deepcopy necessary??
                futures.append(ray_func.remote(X, y, train_idx, test_idx, groups, copy.deepcopy(self.model), strat=strat, n_folds_tuning=self.n_folds_tuning, random_state=self.random_state))
            else:
                futures.append(self.fit_predict_single(X, y, train_idx, test_idx, groups, strat=strat))
                
        if self.use_ray:
            runs = ray.get(futures)
        else:
            runs = futures

        return runs, X, y

def fit_predict_single_ray(X, y, train_idx, test_idx, groups, model, strat=None, n_folds_tuning=5, random_state=2):
    '''Fits, predicts and evaluates one model
    
    This function is necessary as Ray does not accept methods.
    '''
    # train and test sets
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # corresponding group and strat column
    strat_inner = strat.iloc[train_idx]
    groups_inner = groups[train_idx]

    # CV indices for internal hyperparameter tuning by the model
    oof_idx = np.full(y_train.shape, -1)
    sgkf_inner = StratifiedGroupKFold(n_splits=n_folds_tuning, shuffle=True, random_state=random_state)
    for idx_inner, (train_inner, test_inner) in enumerate(sgkf_inner.split(X_train, strat_inner, groups=groups_inner)):
        oof_idx[test_inner] = idx_inner

    # model prediction function
    predictor = model
    predictor.fit(X_train, y_train, oof_idx)
    y_pred = predictor.predict(X_test)
    y_pred = np.array(y_pred).flatten()

    # evaluation
    metrics = [sklearn.metrics.r2_score(y_test, y_pred), sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)]

    # save
    return_dict = {
        #'id': idx,
        'model': predictor,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'pred': y_pred,
        'metrics': metrics
    }
    return return_dict

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
