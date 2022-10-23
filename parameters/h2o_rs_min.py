# Model: H2O
# Expl. variables: RS_min

params = {
    'name': 'H2O-RS_min',
    'desc': 'H2O 600 CPU min (30min limit, 20 CPUs) with RS_min data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': 26,
    'variable_set': 'rs_min',
    'model': 'h2o',
    'model_params': {
        'time': 1*60,
        'scoring': 'RMSE'
    }
}