# Model: RF (Random Search)
# Expl. variables: RS_min

params = {
    'name': 'RF-RS_min',
    'desc': 'Random Forest (Random Search, default parameters) with RS_min data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': None,
    'variable_set': 'rs_min',
    'model': 'random_forest',
    'model_params': {
        'n_cpus': -1
    }
}