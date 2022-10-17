import numpy as np
import sklearn

import random
import copy
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict

class StratifiedGroupKFold():
    '''
    Manual implementation, as not included in sklearn 0.24

    See https://www.kaggle.com/code/jakubwasikowski/stratified-group-k-fold-cross-validation/notebook
    '''
    def __init__(self, n_splits=5, shuffle=True, random_state=2):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        lenc = LabelEncoder()
        y = lenc.fit_transform(y)
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
        
        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

class CV(object):
    '''Runs CV on model
    
    Attributes:
        model (BaseModel): Model instance
        n_folds_cv (int): Number of CV folds (evaluation)
        n_folds_tuning (int): Number of folds for internal CV (training)
        random_state (int): Random state
        num_cpus (int): Number of CPUs
    '''
    def __init__(self, model, n_folds_cv=5, n_folds_tuning=5, random_state=2, num_cpus=None):
        self.model = model
        self.n_folds_cv = n_folds_cv
        self.n_folds_tuning = n_folds_tuning
        self.random_state = random_state
        self.num_cpus = num_cpus

    def fit_predict_single(self, X, y, train_idx, test_idx, groups, strat=None):
        '''Fits, predicts and evaluates a single model
        
        Args:
            X (pd.DataFrame): Data frame with explanatory variables as columns
            y (pd.Series): Target variable
            train_idx (list): Indices of training samples
            test_idx (list): Indices of test samples
            groups (list): Group identifiers
            strat (pd.Series): Series of classes for stratification

        Returns:
            Dictionary with keys model, train_idx, test_idx, pred, metrics
        '''
        # train and test sets
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # corresponding group and strat column
        strat_inner = strat.iloc[train_idx]
        groups_inner = groups[train_idx]

        # CV indices for internal hyperparameter tuning by the model
        oof_idx = np.full(y_train.shape, -1)
        sgkf_inner = StratifiedGroupKFold(n_splits=self.n_folds_tuning, shuffle=True, random_state=self.random_state)
        for idx_inner, (train_inner, test_inner) in enumerate(sgkf_inner.split(X_train, strat_inner, groups=groups_inner)):
            oof_idx[test_inner] = idx_inner

        # model prediction function
        predictor = copy.deepcopy(self.model)
        predictor.fit(X_train, y_train, oof_idx)
        y_pred = predictor.predict(X_test)
        y_pred = np.array(y_pred).flatten()

        # evaluation
        metrics = [sklearn.metrics.r2_score(y_test, y_pred), sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)]

        # save
        return_dict = {
            #'id': idx,
            'model': predictor,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'pred': y_pred,
            'metrics': metrics
        }
        return return_dict
    
    def fit_predict(self, df, y_col, strat=None):
        '''Runs CV loop n_folds_cv times

        Applies fit_predict_single for each CV fold. Groups by first index level of df.
        
        Args:
            df (pd.DataFrame): Data frame with variables as columns, incl. target variable
            y_col (str): Name of target variable
            strat (pd.Series): Series of classes for stratification

        Returns:
            futures: List of fit_predict futures
            X (pd.DataFrame): Data frame with variables as columns, without target variables 
            y (pd.Series): Target variable

        TODO:
            target variable in X?
        '''

        df, strat = self.model.preproc(df, y_col, strat)

        # define groups for outer and inner sampling
        groups = df.index.get_level_values(0).values
        sgkf_outer = StratifiedGroupKFold(n_splits=self.n_folds_cv, shuffle=True, random_state=self.random_state) 

        # predictors
        X = df.drop(y_col, axis=1)
        y = df[y_col]

        futures = []
        for idx, (train_idx, test_idx) in enumerate(sgkf_outer.split(X, strat, groups=groups)): 
            futures.append(self.fit_predict_single(X, y, train_idx, test_idx, groups, strat=strat))

        return futures, X, y