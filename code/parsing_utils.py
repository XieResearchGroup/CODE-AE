import re
import json
import os
import pickle
import numpy as np
import pandas as pd
from operator import itemgetter
from collections import Counter, defaultdict


def get_largest_kv(d, std_dict):
    k = max(d.items(), key=itemgetter(1))[0]
    return k, d[k], std_dict[k]


def parse_param_str(param_str):
    pattern = re.compile('(pretrain_num_epochs)?_?(\d+)?_?(train_num_epochs)_(\d+)_(dop)_(\d\.\d)')
    matches = pattern.findall(param_str)
    return {matches[0][i]: float(matches[0][i + 1]) for i in range(0, len(matches[0]), 2) if matches[0][i] != ''}


def parse_ft_evaluation_result(file_name, method, category, measurement='AUC', metric_name='auroc'):
    folder = f'model_save/{method}/{measurement}/{category}'
    with open(os.path.join(folder, file_name), 'r') as f:
        result_dict = json.load(f)
    return result_dict[metric_name]


def parse_hyper_ft_evaluation_result(method, category, measurement='AUC', metric_name='auroc'):
    folder = f'model_save/{method}/{measurement}/{category}'
    evaluation_metrics = {}
    evaluation_metrics_std = {}
    evaluation_metrics_count = {}
    count = 0
    for file in os.listdir(folder):
        if re.match('(pretrain|train)+.*(dop+).*(ft)+.*\.json', file):
            count += 1
            with open(os.path.join(folder, file), 'r') as f:
                result_dict = json.load(f)
            evaluation_metrics[file] = np.mean(result_dict[metric_name])
            evaluation_metrics_std[file] = np.std(result_dict[metric_name])
            evaluation_metrics_count[file] = len(Counter(result_dict[metric_name])) / len(result_dict[metric_name])
    to_exclude = []
    for k, v in evaluation_metrics_count.items():
        if v < 0.6:
            to_exclude.append(k)

    if len(to_exclude) > 0:
        for k in to_exclude:
            evaluation_metrics.pop(k)
            evaluation_metrics_std.pop(k)

    return evaluation_metrics, evaluation_metrics_std, count


def parse_ml_evaluation_result(file_name, method, category, measurement='AUC', metric_name='auroc'):
    folder = f'model_save/{method}/{measurement}/{category}'
    with open(os.path.join(folder, file_name), 'r') as f:
        result_dict = json.load(f)
    return result_dict['enet'][0][metric_name]


def parse_hyper_ml_evaluation_result(method, category, measurement='AUC', metric_name='auroc'):
    folder = f'model_save/{method}/{category}'
    enet_result = {}
    enet_result_std = {}
    count = 0

    for file in os.listdir(folder):
        if re.match('(pretrain|train)+.*(dop)+.*(ml)+.*', file):
            count += 1
            with open(os.path.join(folder, file), 'r') as f:
                result_dict = json.load(f)
            enet_result[file] = np.mean(result_dict['enet'][0][metric_name])
            enet_result_std[file] = np.std(result_dict['enet'][0][metric_name])

    return enet_result, enet_result_std, count


def generate_hyper_ml_report(metric_name='auroc', measurement='AUC'):
    methods = ['coral', 'adae', 'dsn', 'dsnw', 'code_base', 'code_mmd', 'code_adv']
    categories = ['female', 'male']
    report = pd.DataFrame(np.zeros((len(methods), len(categories))), index=methods, columns=categories)
    report_std = pd.DataFrame(np.zeros((len(methods), len(categories))), index=methods, columns=categories)
    result_dict = dict()

    for cat in categories:
        for method in methods:
            folder = f'model_save/{method}'
            try:
                param_str, report.loc[method, cat], report_std.loc[method, cat] = get_largest_kv(d=
                                                                                                 parse_hyper_ml_evaluation_result(
                                                                                                     method=method,
                                                                                                     category=cat,
                                                                                                     metric_name=metric_name,
                                                                                                     measurement=measurement)[
                                                                                                     0],
                                                                                                 std_dict=
                                                                                                 parse_hyper_ml_evaluation_result(
                                                                                                     method=method,
                                                                                                     category=cat,
                                                                                                     metric_name=metric_name,
                                                                                                     measurement=measurement)[
                                                                                                     1])

                with open(os.path.join(folder, 'train_params.json'), 'r') as f:
                    params_dict = json.load(f)
                result_dict[cat][method] = parse_ml_evaluation_result(file_name=param_str, method=method, category=cat,
                                                                      metric_name=metric_name, measurement=measurement)
                params_dict['unlabeled'].update(parse_param_str(param_str))
                with open(os.path.join(folder, f'train_params_{cat}.json'), 'w') as f:
                    json.dump(params_dict, f)
            except:
                pass

    return report, report_std, result_dict


def generate_hyper_ft_report(metric_name='auroc', measurement='AUC'):
    methods = ['mlp', 'ae', 'dae', 'vae', 'coral', 'adae', 'dsn', 'dsnw', 'code_base', 'code_mmd', 'code_adv']
    categories = ['gem', 'fu', 'cis', 'tem']
    report = pd.DataFrame(np.zeros((len(methods), len(categories))), index=methods, columns=categories)
    report_std = pd.DataFrame(np.zeros((len(methods), len(categories))), index=methods, columns=categories)
    result_dict = defaultdict(dict)

    for cat in categories:
        for method in methods:
            folder = f'model_save/{method}'
            try:
                param_str, report.loc[method, cat], report_std.loc[method, cat] = get_largest_kv(d=
                                                                                                 parse_hyper_ft_evaluation_result(
                                                                                                     method=method,
                                                                                                     category=cat,
                                                                                                     metric_name=metric_name,
                                                                                                     measurement=measurement)[
                                                                                                     0],
                                                                                                 std_dict=
                                                                                                 parse_hyper_ft_evaluation_result(
                                                                                                     method=method,
                                                                                                     category=cat,
                                                                                                     metric_name=metric_name,
                                                                                                     measurement=measurement)[
                                                                                                     1])
                with open(os.path.join(folder, 'train_params.json'), 'r') as f:
                    params_dict = json.load(f)
                result_dict[cat][method] = parse_ft_evaluation_result(file_name=param_str, method=method, category=cat,
                                                                      metric_name=metric_name, measurement=measurement)
                params_dict['unlabeled'].update(parse_param_str(param_str))
                with open(os.path.join(folder, f'train_params_{cat}.json'), 'w') as f:
                    json.dump(params_dict, f)
            except:
                pass

    return report, report_std, result_dict


def load_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return data
