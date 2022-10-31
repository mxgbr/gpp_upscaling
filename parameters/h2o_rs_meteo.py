# Model: H2O
# Expl. variables: RS meteo

params = {
    'name': 'H2O-RS_meteo',
    'desc': 'H2O 600 CPU min (30min limit, 20 CPUs) with RS meteo data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': 26,
    'variable_set': 'rs_meteo',
    'model': 'h2o',
    'model_params': {
        'time': 30*60,
        'scoring': 'RMSE'
    }
}