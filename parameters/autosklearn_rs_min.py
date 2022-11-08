# Model: AutoSklearn
# Expl. variables: RS_min

params = {
    'name': 'AutoSklearn-RS_min',
    'desc': 'AutoSklearn with RS minimal data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': 26,
    'variable_set': 'rs_min',
    'model': 'autosklearn',
    'model_params': {
        'time': 30*60,
        'n_cpus': 20,
        'scoring': 'mean_squared_error'
    }
}