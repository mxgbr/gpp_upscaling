# Model: AutoSklearn
# Expl. variables: RS

params = {
    'name': 'AutoSklearn-RS',
    'desc': 'AutoSklearn with RS data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'bootstrap_size': 0.8,
    'random_state': 26,
    'variable_set': 'rs',
    'model': 'autosklearn',
    'model_params': {
        'time': 30*60,
        'n_cpus': 20,
        'scoring': 'mean_squared_error'
    }
}