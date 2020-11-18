# import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import argparse
import os
import data_config


def roc_auc_scorer(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def classify_with_rf(train_features, y_train, cv_split_rf):
    try:
        # logger.debug("Training Random Forest model")
        # mx_depth: trees' maximum depth
        # n_estimators: number of trees to use
        # n_jobs = -1 means to run the jobs in parallel
        rf_tuning_parameters = [{'n_estimators': [10, 50, 200, 500], 'max_depth': [10, 20, 50, 100, 200]}]
        # rf_tuning_parameters = [{'n_estimators': [5], 'max_depth': [10]}]
        rf = GridSearchCV(RandomForestClassifier(), rf_tuning_parameters, n_jobs=-1, cv=cv_split_rf,
                          verbose=2, scoring=make_scorer(roc_auc_scorer, needs_proba=True))
        rf.fit(train_features, y_train)  # , groups=train_groups
        # logger.debug("Trained Random Forest successfully")
        return rf

    except Exception as e:
        # logger.debug("Fail to Train Random Forest, caused by %s" % e.message)
        raise e


def regress_with_rf(train_features, y_train, cv_split_rf):
    try:
        # logger.debug("Training Random Forest model")
        # mx_depth: trees' maximum depth
        # n_estimators: number of trees to use
        # n_jobs = -1 means to run the jobs in parallel
        rf_tuning_parameters = [{'n_estimators': [10, 50, 200, 500], 'max_depth': [10, 20, 50, 100, 200]}]
        # rf_tuning_parameters = [{'n_estimators': [5], 'max_depth': [10]}]
        rf = GridSearchCV(RandomForestRegressor(), rf_tuning_parameters, n_jobs=-1, cv=cv_split_rf,
                          verbose=2)
        rf.fit(train_features, y_train)  # , groups=train_groups
        # logger.debug("Trained Random Forest successfully")
        return rf

    except Exception as e:
        # logger.debug("Fail to Train Random Forest, caused by %s" % e.message)
        raise e


def classify_with_enet(train_features, y_train, cv_split_enet):
    try:
        # logger.debug("Training elastic net regression model")
        alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        # alphas = [0.1]
        # l1_ratios = [0.25]
        base_enet = SGDClassifier(loss='log', penalty='elasticnet', random_state=2020)
        enet_param_grid = dict(alpha=alphas, l1_ratio=l1_ratios)
        enet = GridSearchCV(estimator=base_enet, param_grid=enet_param_grid, n_jobs=-1, cv=cv_split_enet, verbose=2,
                            scoring=make_scorer(roc_auc_scorer, needs_proba=True))
        enet.fit(train_features, y_train)
        # logger.debug("Trained Elastic net classification model successfully")
        return enet
    except Exception as e:
        raise e


def regress_with_enet(train_features, y_train, cv_split_enet):
    try:
        # logger.debug("Training elastic net regression model")
        l1_ratios = np.logspace(-5, -1, num=5, endpoint=True)
        alphas = np.array([0.1, 1.0, 10])
        enet = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=cv_split_enet, n_jobs=-1, verbose=2,
                            normalize=True,
                            max_iter=10000)
        enet.fit(train_features, y_train)
        # logger.debug("Trained Elastic net regression model successfully")
        return enet
    except Exception as e:
        raise e


def classify_kfold_prediction(feature_df, target_df, method='rf'):
    assert all(feature_df.index == target_df.index)
    pred_df = pd.DataFrame(np.full_like(target_df.values, fill_value=-1), index=target_df.index,
                           columns=target_df.columns)
    if method == 'enet':
        model_fn = classify_with_enet
    else:
        model_fn = classify_with_rf

    for drug in target_df.columns:
        print(f'Drug: {drug}')
        y = target_df.loc[~target_df[drug].isna(), drug].astype('int')
        X = feature_df.loc[y.index]
        assert all(y.index == X.index)

        outer_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
        for train_index, test_index in outer_kfold.split(X, y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=2020)
            cv_split = kfold.split(X_train, y_train)
            try:
                trained_model = model_fn(X_train, y_train, list(cv_split))
                prediction = trained_model.predict_proba(X_test)[:, 1]
                pred_df.loc[y.index[test_index], drug] = prediction
            except Exception as e:
                print(e)
        pred_df.to_csv(f'{method}_pred.csv', index_label='Sample')


if __name__ == '__main__':
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)
    target_df = pd.read_csv(data_config.gdsc_preprocessed_target_file, index_col=0)
    target_samples = list(
        set(gex_features_df.index.to_list()) & set(target_df.index.to_list()))
    labeled_gex_df = gex_features_df.loc[target_samples, :]
    target_df = target_df.loc[target_samples, :]
    target_df = target_df[target_df.columns[target_df.sum() >= 10]]
    target_df = target_df[target_df.isna().sum().sort_values().index]
    classify_kfold_prediction(feature_df=labeled_gex_df, target_df=target_df, method='enet')
