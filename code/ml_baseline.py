from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score, precision_recall_curve, accuracy_score, \
    f1_score, auc
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import random
from collections import defaultdict
from evaluation_utils import evaluate_target_classification_epoch, model_save_check
from itertools import chain
import os
import pandas as pd
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import itertools
import numpy as np
import data
import data_config


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')


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
        kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
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


def main(args, drug):
    if args.method == 'rf':
        model_fn = classify_with_rf
    else:
        model_fn = classify_with_enet

    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)

    if args.pdtc_flag:
        task_save_folder = os.path.join('model_save', args.method, args.measurement, 'pdtc', drug)
    else:
        task_save_folder = os.path.join('model_save', args.method, args.measurement, drug)

    safe_make_dir(task_save_folder)

    random.seed(2020)
    ft_evaluation_metrics = defaultdict(list)

    labeled_ccle_dataloader, labeled_tcga_dataloader = data.get_labeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=64,
        drug=drug,
        ft_flag=False,
        ccle_measurement=args.measurement,
        threshold=None,
        days_threshold=None,
        pdtc_flag=args.pdtc_flag
    )

    metric_result_list = n_time_cv(
        model_fn=model_fn,
        n=args.n,
        train_data=(
            labeled_ccle_dataloader.dataset.tensors[0].numpy(),
            labeled_ccle_dataloader.dataset.tensors[1].numpy()
        ),
        test_data=(
            labeled_tcga_dataloader.dataset.tensors[0].numpy(),
            labeled_tcga_dataloader.dataset.tensors[1].numpy()
        ),
        metric=args.metric
    )


    with open(os.path.join(task_save_folder, f'ft_evaluation_results.json'), 'w') as f:
        json.dump(dict(metric_result_list[1]), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ML training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='enet', choices=['enet', 'rf'])
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--a_thres', dest='a_thres', nargs='?', type=float, default=None)
    parser.add_argument('--d_thres', dest='days_thres', nargs='?', type=float, default=None)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    args = parser.parse_args()

    if args.pdtc_flag:
        drug_list = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0).index.tolist()
    else:
        drug_list = ['tgem', 'tfu', 'tem', 'gem', 'cis', 'sor', 'fu']
    for drug in drug_list:
        main(args=args, drug=drug)
