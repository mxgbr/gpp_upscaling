# Model: AutoSklearn
# Expl. variables: RS_min_vi

params = {
    'name': 'AutoSklearn-RS_min_vi',
    'desc': 'AutoSklearn with RS minimal +VI data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': 26,
    'variable_set': 'rs_min_vi',
    'model': 'autosklearn',
    'model_params': {
        'time': 30*60,
        'n_cpus': 20,
        'scoring': 'mean_squared_error'
    }
}