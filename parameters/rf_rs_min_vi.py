# Model: RF (Random Search)
# Expl. variables: RS_min_vi

params = {
    'name': 'RF-RS_min_vi',
    'desc': 'Random Forest (Random Search, default parameters) with RS_min_vi data set',
    'n_folds_cv': 5,
    'n_folds_tuning': 5,
    'random_state': 26,
    'variable_set': 'rs_min_vi',
    'model': 'random_forest',
    'model_params': {
        'n_estimators': [100, 200, 400, 800, 1600],
        'max_features': 15,
        'min_samples_leaf': [2, 5, 10],
        'min_samples_split': [4, 8, 12, 16],
        'max_depth': [5, 10, 20, None],
        'n_iter': 125,
        'n_cpus': 20,
        'scoring': 'neg_root_mean_squared_error'
    }
}