# Model: H2O
# Expl. variables: RS_yr

params = {
    'name': 'H2O-RS_yr',
    'desc': 'H2O 600 CPU min (30min limit, 20 CPUs) with RS_yr data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': 26,
    'variable_set': 'rs_yr',
    'model': 'h2o',
    'model_params': {
        'time': 30*60,
        'scoring': 'RMSE'
    }
}