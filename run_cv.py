import pandas as pd
import sys
import modules.utils as utils
from modules.cv import CV
import importlib.util

if __name__ == "__main__":

    # check if array task
    array_job = len(sys.argv) > 2

    if array_job:
        array_id = sys.argv[2]
    else:
        array_id = None

    # load parameters from file
    params_file = sys.argv[1]
    spec = importlib.util.spec_from_file_location("module.name", params_file)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    params = params.params

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

    else:
        raise AttributeError('Invalid model choice')  

    # load data
    y_var = 'GPP_NT_CUT_REF'
    data = pd.read_csv('data/ec/data_monthly_0_05_2001-2020_v1.csv', index_col=['SITE_ID', 'Date'], parse_dates=True) 
    data_sel = utils.preprocess(data, params['variable_set'], target=[y_var,'MODIS_LC'], cat=['MODIS_LC'])
    strat = data_sel['MODIS_LC']
    data_sel = data_sel.drop('MODIS_LC', axis=1)

    # pre-processing
    if params['model'] == 'random_forest':
       data_sel, strat = preproc(data_sel, strat)

    # set up experiment
    exp = utils.Experiment(suffix=array_id)

    # run model
    model = ModelWrapper(**params['model_params'])
    cv = CV(model, 
                n_folds_cv=params['n_folds_cv'], 
                n_folds_tuning=params['n_folds_tuning'],
                random_state=params['random_state'],
                num_cpus=params['num_cpus'])

    runs, X, y = cv.fit_predict(data_sel, y_var, strat)

    runs = {k: [dic[k] for dic in runs] for k in runs[0]}

    # save results
    exp.save(len(runs), X=X, y=y, params=params, models=runs['model'], train_idx=runs['train_idx'], test_idx=runs['test_idx'], y_pred=runs['pred'])

    print('PYTHON DONE')
    quit()