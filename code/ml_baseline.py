from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score, precision_recall_curve, accuracy_score, \
    f1_score, auc
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import random
from collections import defaultdict


def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)


scoring = {
    'auroc': 'roc_auc',
    'auprc': make_scorer(auprc, needs_proba=True),
    'acc': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'aps': make_scorer(average_precision_score, needs_proba=True)
}


def classify_with_rf(train_features, y_train, cv_split_rf, metric='auroc'):
    try:
        # logger.debug("Training Random Forest model")
        # mx_depth: trees' maximum depth
        # n_estimators: number of trees to use
        # n_jobs = -1 means to run the jobs in parallel
        rf_tuning_parameters = [{'n_estimators': [10, 50, 200, 500, 1000], 'max_depth': [10, 50, 100, 200, 500]}]
        # rf_tuning_parameters = [{'n_estimators': [5], 'max_depth': [10]}]
        rf = GridSearchCV(RandomForestClassifier(), rf_tuning_parameters, n_jobs=-1, cv=cv_split_rf,
                          verbose=2, scoring=scoring, refit=metric)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        rf.fit(train_features, y_train)  # , groups=train_groups
        # logger.debug("Trained Random Forest successfully")
        return rf, scaler

    except Exception as e:
        # logger.debug("Fail to Train Random Forest, caused by %s" % e.message)
        raise e


def classify_with_enet(train_features, y_train, cv_split_enet, metric='auroc'):
    try:
        # logger.debug("Training elastic net regression model")
        alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        # alphas = [0.1]
        # l1_ratios = [0.25]
        base_enet = SGDClassifier(loss='log', penalty='elasticnet', random_state=12345)
        enet_param_grid = dict(alpha=alphas, l1_ratio=l1_ratios)
        enet = GridSearchCV(estimator=base_enet, param_grid=enet_param_grid, n_jobs=-1, cv=cv_split_enet, verbose=2,
                            scoring=scoring, refit=metric)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        enet.fit(train_features, y_train)
        # logger.debug("Trained Elastic net classification model successfully")
        return enet, scaler
    except Exception as e:
        raise e


def n_time_cv(train_data, n=10, model_fn=classify_with_enet, test_data=None, random_state=2020, metric='auroc'):
    # metric_list = ['auroc', 'acc', 'aps', 'f1']
    metric_list = ['auroc', 'acc', 'aps', 'f1', 'auprc']

    random.seed(random_state)
    seeds = random.sample(range(100000), k=n)
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    models = []
    for seed in seeds:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_split = kfold.split(*train_data)
        trained_model, scaler = model_fn(*train_data, list(cv_split), metric=metric)
        for metric in metric_list:
            train_history[metric].append(trained_model.cv_results_[f'mean_test_{metric}'][trained_model.best_index_])
        if test_data is not None:
            # preds = trained_model.predict(test_data[0])
            # pred_scores = trained_model.predict_proba(test_data[0])[:, 1]
            preds = trained_model.predict(scaler.transform(test_data[0]))
            pred_scores = trained_model.predict_proba(scaler.transform(test_data[0]))[:, 1]

            # print(preds)
            # print(pred_scores)
            test_history['auroc'].append(roc_auc_score(y_true=test_data[1], y_score=pred_scores))
            test_history['acc'].append(accuracy_score(y_true=test_data[1], y_pred=preds))
            test_history['aps'].append(average_precision_score(y_true=test_data[1], y_score=pred_scores))
            test_history['f1'].append(f1_score(y_true=test_data[1], y_pred=preds))
            test_history['auprc'].append(auprc(y_true=test_data[1], y_score=pred_scores))

        models.append(trained_model)

    return (train_history, models) if test_data is None else (train_history, test_history, models)
