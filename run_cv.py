# This file performs a CV for the models, parameters, and variables specified in the provided parameter file.
# The CV is stratified for MODIS_LC and has GPP_NT_CUT_REF as target variable
# The CV can be repeated with SLURM array jobs
#
# 2 arguments required: path of parameter file and $SLURM_JOB_ID or $SLURM_ARRAY_JOB_ID
# 1 argument optional: $SLURM_ARRAY_TASK_ID
# Saves experiment results in experiments/<Date><$SLURM_ARRAY_JOB_ID>_<$SLURM_ARRAY_TASK_ID>

import pandas as pd
import sys
import modules.utils as utils
from modules.training import CV
import importlib.util

if __name__ == "__main__":

    # check if array task
    array_job = len(sys.argv) > 3

    slurm_id = sys.argv[2]

    if array_job:
        array_id = sys.argv[3]
    else:
        array_id = None

    # load parameters from file
    params_file = sys.argv[1]
    spec = importlib.util.spec_from_file_location("module.name", params_file)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    params = params.params

    # random state, change with array id to make splits different but reproducible
    if array_job & (params['random_state'] is not None):
        params['random_state'] = params['random_state'] + int(array_id)

    # select model
    if params['model'] == 'random_forest':
        from models.rf import RandomForestCV as ModelWrapper

        def preproc(data_sel, strat):
            '''Fills gaps in data

            RF cannot handle missing data. This function fills gaps with the avg of their neighboring values (within a site). 
            Data is removed where filling is impossible.

            Args:
                data_sel (pd.DataFrame): Data frame with SITE_ID as index level
                strat (pd.Series): Series of strat variable

            Returns:
                data_sel (pd.DataFrame): Cleaned df
                strat (pd.Series): Cleaned strat series
            '''
            data_sel = data_sel.groupby('SITE_ID').transform(lambda x: x.where(x.notna(), other=(x.fillna(method='ffill') + x.fillna(method='bfill'))/2))

            # drop rows that could not be filled
            na_mask = data_sel.notna().all(axis=1)
            data_sel = data_sel[na_mask]
            strat = strat[na_mask]

            return data_sel, strat

    elif params['model'] == 'h2o':
        from models.h2o import H2o as ModelWrapper

    elif params['model'] == 'autosklearn':
        from models.autosklearn import AutoSklearn as ModelWrapper

    elif params['model'] == 'autogluon':
        from models.autogluon import AutoGluon as ModelWrapper

    else:
        raise AttributeError('Invalid model choice')  

    # load data
    y_var = 'GPP_NT_CUT_REF'
    # data/ec/data_monthly_0_05_2001-2020_v1.csv
    # data/ec/data_monthly_500_v4.csv
    data = pd.read_csv('data/ec/data_monthly_0_05_2001-2020_v1.csv', index_col=['SITE_ID', 'Date'], parse_dates=True) 
    data_sel, target = utils.preprocess(data, params['variable_set'], target=[y_var,'MODIS_LC'], cat=['MODIS_LC'])
    strat = target['MODIS_LC']
    data_sel[y_var] = target[y_var]

    # pre-processing
    if params['model'] == 'random_forest':
       data_sel, strat = preproc(data_sel, strat)

    # set up experiment
    exp = utils.Experiment(exp_id=slurm_id, suffix=array_id, logging=True)

    print('Using explanatory variables:')
    print(list(data_sel.columns))

    # run model
    model = ModelWrapper(**params['model_params'])
    cv = CV(model, 
                n_folds_cv=params['n_folds_cv'], 
                n_folds_tuning=params['n_folds_tuning'],
                random_state=params['random_state'])

    runs, X, y = cv.run(data_sel, y_var, strat)

    runs = {k: [dic[k] for dic in runs] for k in runs[0]}

    print(runs['metrics'])

    # save results
    exp.save(len(runs['model']), X=X, y=y, params=params, models=runs['model'], train_idx=runs['train_idx'], test_idx=runs['test_idx'], y_pred=runs['pred'])

    print('PYTHON DONE')
    quit()