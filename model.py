from sklearn.model_selection import ParameterGrid
from collections import defaultdict
from datetime import datetime
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class LGBMTuning:
    """
    Tuning and RFE class specific for LightGBM
    """
    def __init__(self):
        self.param_grid = {
            'objective': ['binary'],
            'boosting': ['gbdt'],
            'n_estimator': [2500],
            'num_leaves': [30, 50, 70, 90],
            'min_child_sample': [50, 150, 300],
            'min_child_weight': [1, 5, 10, 15],
            'max_depth': list(range(7, 16, 2)),
            'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
            'reg_alpha': [0, 2, 10, 50],
            'reg_lambda': [0, 5, 20, 100],
            'colsample_bytree': [0.7, 0.8],
            'max_bin': [255, 355]
        }
        self.param_combs = None
        self.search_result = None
        self.rfe_result = None
        self.rfe_feature_list = None

    def lgb_randomized_search_cv(self, X, y, metrics, weight=None, folds=None,
                                 stratified=True, nfold=3, n_comb=20, random_state=123,
                                 early_stopping_rounds=5, **kwargs):
        """

        :param X: predictors
        :param y: target
        :param metrics: 'auc', 'logloss' etc
        :param weight: sample weights
        :param folds: presplit folds
        :param stratified: whether to do stratified sampling in cv
        :param nfold: number of folds
        :param n_comb: number of combinations to test
        :param random_state: random_state
        :param early_stopping_rounds: early_stopping_rounds
        :param kwargs:
        :return: None
        """
        np.random.seed(random_state)
        param_combs = np.random.choice(list(ParameterGrid(self.param_grid)),
                                       n_comb, replace=False)

        search_results = defaultdict(list)

        # Ad-hoc treatment;
        if pd.Series(y).nunique() > 10:
            msg = 'Regression task, will use KFold instead of StratifiedKFold'
            print(msg)
            stratified = False

        for idx, param_comb in enumerate(param_combs):
            begin = datetime.now()

            del param_comb['n_estimator']

            train_set = lgb.Dataset(X, label=y, weight=weight)
            cv = lgb.cv(params=param_comb,
                        train_set=train_set,
                        num_boost_round=self.param_grid['n_estimator'][0],
                        folds=folds,
                        stratified=stratified,
                        nfold=nfold,
                        metrics=metrics,
                        seed=random_state,
                        eval_train_metric=True,
                        **kwargs)

            search_results['param_comb'].append(
                {'early_stopping_rounds': early_stopping_rounds, **param_comb}
            )
            search_results[f'mean_train_{metrics}'].append(cv[f'train {metrics}-mean'][-1])
            search_results[f'mean_valid_{metrics}'].append(cv[f'valid {metrics}-mean'][-1])
            search_results['best_n_estimators'].append(len(cv[f'train {metrics}-mean']))

            end = datetime.now()

            msg = 'Finished {idx}/{n_comb} cv with mean valid {metrics}: {score:6.4f} -- {time}s'
            print(msg.format(idx=idx+1, n_comb=n_comb, metrics=metrics,
                             score=search_results[f'mean_valid_{metrics}'][-1],
                             time=(end-begin).seconds))

        self.param_combs = search_results['param_comb']
        self.search_result = pd.DataFrame(
            {'model_idx': range(n_comb),
             f'mean_train_{metrics}': search_results[f'mean_train_{metrics}'],
             f'mean_valid_{metrics}': search_results[f'mean_valid_{metrics}'],
             'best_n_estimators': search_results['best_n_estimators']}
        )
        self.search_result['train_valid_diff'] = \
            self.search_result[f'mean_train_{metrics}'] - \
            self.search_result[f'mean_valid_{metrics}']

    def lgb_rfe(self, X, y, metrics, param=0, weight=None,
                folds=None, stratified=True, nfold=3, step=25,
                min_features=150, early_stopping_rounds=5,
                random_state=123, **kwargs):
        """

        :param X: predictors
        :param y: target
        :param metrics: 'auc', 'logloss' etc.
        :param param: integer for model index, or dictionary of parameters
        :param weight: sample weights
        :param folds: presplit folds
        :param stratified: whether to do stratified sampling
        :param nfold: number of cv folds
        :param step: number features to eliminate each iter
        :param min_features: below which the process will stop
        :param early_stopping_rounds: early_stopping_rounds
        :param random_state: random_state
        :param kwargs:
        :return: None
        """
        x = X.copy()

        if isinstance(param, int):
            model_idx = param
            param_comb = self.search_result[model_idx].copy()
        elif isinstance(param, dict):
            param_comb = param.copy()
            model_idx = -1
        else:
            raise ValueError('Param only takes dict and integer.')

        rfe_result = defaultdict(list)

        if pd.Series(y).nunique() > 10:
            msg = 'Regression task, will use KFold instead of StratifiedKFold'
            print(msg)
            stratified = False

        del param_comb['n_estimator']

        while x.shape[1] >= min_features:
            begin = datetime.now()

            train_set = lgb.Dataset(x, label=y, weight=weight)
            cv = lgb.cv(params=param_comb,
                        train_set=train_set,
                        folds=folds,
                        stratified=stratified,
                        num_boost_round=self.param_grid['n_estimator'][0],
                        early_stopping_rounds=early_stopping_rounds,
                        nfold=nfold, metrics=metrics, seed=random_state,
                        eval_train_metric=True, **kwargs)
            rfe_result['n_features'].append(x.shape[1])
            rfe_result[f'mean_train_{metrics}'].append(cv[f'train {metrics}-mean'][-1])
            rfe_result[f'mean_valid_{metrics}'].append(cv[f'valid {metrics}-mean'][-1])
            rfe_result['best_n_estimators'].append(len(cv[f'train {metrics}-mean']))
            rfe_result['features'].append(list(x.columns))

            model = lgb.train(params=param_comb, train_set=train_set,
                              num_boost_round=rfe_result['best_n_estimators'][-1],
                              verbose_eval=False)
            importance = pd.DataFrame(
                {'feature': x.columns.values,
                 # ad-hoc define importance metric
                 'importance': model.feature_importance(importance_type='gain')}
            )
            importance['rank'] = importance['importance'].rank(ascending=False)

            to_drop = importance.nsmallest(step, 'importance').loc[:, 'feature']
            x.drop(to_drop, axis=1, inplace=True)

            end = datetime.now()

            msg = '{n_features} with mean valid {metrics}: {score:6.4f} -- {time}s'
            print(msg.format(n_features=rfe_result['n_features'][-1],
                             metrics=metrics,
                             score=rfe_result[f'mean_valid_{metrics}'][-1],
                             time=(end - begin).seconds))

        self.rfe_result = pd.DataFrame(
            {'model_idx': model_idx,
             'n_features': rfe_result['n_features'],
             f'mean_train_{metrics}': rfe_result[f'mean_train_{metrics}'],
             f'mean_valid_{metrics}': rfe_result[f'mean_valid_{metrics}'],
             'best_n_estimators': rfe_result['best_n_estimators']}
        )
        self.rfe_feature_list = rfe_result['features']

    def plot_rfe(self):

        if self.rfe_result is None:
            raise AttributeError('RFE not run or result not given.')

        fig, axes = plt.subplots(2, 1, figsize=(18, 16))
        for col_idx, ax in zip([2, 3], axes):
            col_label = self.rfe_result.columns.values[col_idx].upper()
            y = self.rfe_result.iloc[:, col_idx]

            color = 'C0-D' if col_idx == 2 else 'C1-D'

            ax.plot(self.rfe_result['n_features'], y, color, label=col_label)
            ax.set_xticks(self.rfe_result['n_features'])
            ax.legend()

            for x, y in zip(self.rfe_result['n_features'],
                            self.rfe_result.iloc[:, col_idx]):
                label = '{:.4f}'.format(y)
                ax.annotate(label, (x, y),
                            textcoords='offset points',
                            xytext=(0, 10), ha='center')
            ax.set_xlim([self.rfe_result['n_features'].max()*1.05,
                         self.rfe_result['n_features'].min()*0.9])
            ax.set_ylim([self.rfe_result.iloc[:, col_idx].min() -
                         0.5*self.rfe_result.iloc[:, col_idx].std(),
                         self.rfe_result.iloc[:, col_idx].max() +
                         0.5*self.rfe_result.iloc[:, col_idx].std()])
            ax.set_xlabel('Number of Features', size=12)
            ax.set_ylabel(col_label, size=12)
            ax.set_title(f'{col_label} vs no.features', size=14)

        plt.show()
