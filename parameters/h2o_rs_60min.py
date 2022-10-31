# Model: H2O
# Expl. variables: RS
# Runtime: 60min (1200 CPU min)

params = {
    'name': 'H2O-RS',
    'desc': 'H2O 1200 CPU min (60min limit, 20 CPUs) with RS data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': 26,
    'variable_set': 'rs',
    'model': 'h2o',
    'model_params': {
        'time': 60*60,
        'scoring': 'RMSE'
    }
}