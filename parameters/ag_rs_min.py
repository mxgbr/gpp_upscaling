# Model: AutoGluon
# Expl. variables: RS_min

params = {
    'name': 'AutoGluon-RS_min',
    'desc': 'AutoGluon with RS_min data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': 26,
    'variable_set': 'rs_min',
    'model': 'autogluon',
    'model_params': {
        'time': 30*60,
        'scoring': 'root_mean_squared_error',
        'bag_sets': 20, #default
        'stack_levels': 3, #recommended 1-3 for superior performance
        'bag_folds': 5 #recommended 5-10 for superior performance
    }
}