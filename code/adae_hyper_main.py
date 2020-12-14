import pandas as pd
import torch
import json
import os
import argparse
import pickle
import itertools
import random
from collections import defaultdict
from copy import deepcopy

import data
import data_config
import train_ndsn
import train_adae
import train_adsn
import train_coral
import train_dae
import train_vae
import train_ae
import train_mdsn
import train_dsnw
import train_dsn

import fine_tuning
import ml_baseline


def generate_encoded_features(encoder, dataloader, normalize_flag=False):
    """

    :param normalize_flag:
    :param encoder:
    :param dataloader:
    :return:
    """
    encoder.eval()
    raw_feature_tensor = dataloader.dataset.tensors[0].cpu()
    label_tensor = dataloader.dataset.tensors[1].cpu()

    encoded_feature_tensor = encoder.cpu()(raw_feature_tensor)
    if normalize_flag:
        encoded_feature_tensor = torch.nn.functional.normalize(encoded_feature_tensor, p=2, dim=1)
    return encoded_feature_tensor, label_tensor


def load_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return data


def wrap_training_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])

    return aux_dict


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


def main(args, update_params_dict):
    if args.method == 'dsn':
        train_fn = train_dsn.train_dsn
    elif args.method == 'adae':
        train_fn = train_adae.train_adae
    elif args.method == 'coral':
        train_fn = train_coral.train_coral
    elif args.method == 'dae':
        train_fn = train_dae.train_dae
    elif args.method == 'vae':
        train_fn = train_vae.train_vae
    elif args.method == 'ae':
        train_fn = train_ae.train_ae
    elif args.method == 'mdsn':
        train_fn = train_mdsn.train_mdsn
    elif args.method == 'ndsn':
        train_fn = train_ndsn.train_ndsn
    elif args.method == 'dsnw':
        train_fn = train_dsnw.train_dsnw
    else:
        train_fn = train_adsn.train_adsn

    normalize_flag = args.method in ['adsn', 'mdsn', 'ndsn']
    # normalize_flag = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gex_features_df = pd.read_csv(data_config.adae_gex_file, sep='\t', index_col=0)

    with open(os.path.join('model_save', args.method, 'train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            'model_save_folder': os.path.join('model_save', args.method, args.gender),
            'es_flag': False
        })

    training_params['unlabeled'].update(update_params_dict)

    param_str = dict_to_str(update_params_dict)
    safe_make_dir(training_params['model_save_folder'])

    ml_baseline_history = defaultdict(list)
    random.seed(2020)
    seeds = random.sample(range(100000), k=int(args.n))

    s_dataloaders, t_dataloaders = data.get_adae_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=training_params['unlabeled']['batch_size']
    )

    labeled_pos_dataloader, labeled_neg_dataloader = data.get_adae_labeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=training_params['labeled']['batch_size'],
        pos_gender=args.gender
    )

    # start unlabeled training
    encoder, historys = train_fn(s_dataloaders=s_dataloaders,
                                 t_dataloaders=t_dataloaders,
                                 **wrap_training_params(training_params, type='unlabeled'))

    with open(os.path.join(training_params['model_save_folder'], f'{param_str}_unlabel_train_history.pickle'),
              'wb') as f:
        for history in historys:
            pickle.dump(dict(history), f)

    # generate encoded features
    pos_encoded_feature_tensor, pos_label_tensor = generate_encoded_features(encoder, labeled_pos_dataloader,
                                                                             normalize_flag=normalize_flag)
    neg_encoded_feature_tensor, neg_label_tensor = generate_encoded_features(encoder, labeled_neg_dataloader,
                                                                             normalize_flag=normalize_flag)

    pd.DataFrame(pos_encoded_feature_tensor.detach().cpu().numpy()).to_csv(
        os.path.join(training_params['model_save_folder'], f'{param_str}_train_encoded_feature.csv'))
    pd.DataFrame(pos_label_tensor.detach().cpu().numpy()).to_csv(
        os.path.join(training_params['model_save_folder'], f'{param_str}_train_label.csv'))
    pd.DataFrame(neg_encoded_feature_tensor.detach().cpu().numpy()).to_csv(
        os.path.join(training_params['model_save_folder'], f'{param_str}_test_encoded_feature.csv'))
    pd.DataFrame(neg_label_tensor.detach().cpu().numpy()).to_csv(
        os.path.join(training_params['model_save_folder'], f'{param_str}_test_label.csv'))
    # build baseline ml models for encoded features
    # ml_baseline_history['rf'].append(
    #     ml_baseline.n_time_cv(
    #         model_fn=ml_baseline.classify_with_rf,
    #         n=args.n,
    #         train_data=(
    #             pos_encoded_feature_tensor.detach().cpu().numpy(),
    #             pos_label_tensor.detach().cpu().numpy()
    #         ),
    #         test_data=(
    #             neg_encoded_feature_tensor.detach().cpu().numpy(),
    #             neg_label_tensor.detach().cpu().numpy()
    #         )
    #     )[1]
    # )

    ml_baseline_history['enet'].append(
        ml_baseline.n_time_cv(
            model_fn=ml_baseline.classify_with_enet,
            n=args.n,
            train_data=(
                pos_encoded_feature_tensor.detach().cpu().numpy(),
                pos_label_tensor.detach().cpu().numpy()
            ),
            test_data=(
                neg_encoded_feature_tensor.detach().cpu().numpy(),
                neg_label_tensor.detach().cpu().numpy()
            ),
            metric='auprc'
        )[1]
    )
    with open(os.path.join(training_params['model_save_folder'], f'{param_str}_ml_baseline_results.json'), 'w') as f:
        json.dump(ml_baseline_history, f)

    # start fine-tuning encoder
    # target_classifier, ft_historys = fine_tuning.fine_tune_encoder(
    #     encoder=encoder,
    #     train_dataloader=labeled_pos_dataloader,
    #     val_dataloader=labeled_neg_dataloader,
    #     test_dataloader=labeled_neg_dataloader,
    #     normalize_flag=normalize_flag,
    #     **wrap_training_params(training_params, type='labeled')
    # )
    #
    # with open(os.path.join(training_params['model_save_folder'], f'{param_str}_ft_train_history.pickle'), 'wb') as f:
    #     for history in ft_historys:
    #         pickle.dump(dict(history), f)
    ft_evaluation_metrics = defaultdict(list)
    for seed in seeds:
        train_labeled_pos_dataloader, val_labeled_pos_dataloader, labeled_neg_dataloader = data.get_adae_labeled_dataloaders(
            gex_features_df=gex_features_df,
            seed=seed,
            batch_size=training_params['labeled']['batch_size'],
            pos_gender=args.gender,
            ft_flag=True
        )
        # start fine-tuning encoder
        ft_encoder = deepcopy(encoder)

        target_classifier, ft_historys = fine_tuning.fine_tune_encoder(
            encoder=ft_encoder,
            train_dataloader=train_labeled_pos_dataloader,
            val_dataloader=val_labeled_pos_dataloader,
            test_dataloader=labeled_neg_dataloader,
            normalize_flag=normalize_flag,
            metric_name='auprc',
            **wrap_training_params(training_params, type='labeled')
        )

        # with open(os.path.join(training_params['model_save_folder'], f'ft_train_history_{seed}.pickle'),
        #           'wb') as f:
        #     for history in ft_historys:
        #         pickle.dump(dict(history), f)

        for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
            ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])

    with open(os.path.join(training_params['model_save_folder'], f'{param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(ft_evaluation_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='adsn',
                        choices=['adsn', 'dsn', 'ndsn', 'mdsn', 'dsnw', 'adae', 'coral', 'dae', 'vae', 'ae'])
    parser.add_argument('--gender', dest='gender', nargs='?', default='female', choices=['female', 'male'])
    parser.add_argument('--n', dest='n', nargs='?', default=10)

    args = parser.parse_args()

    params_grid = {
        "pretrain_num_epochs": [0, 50, 100, 200, 300],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
        "dop": [0.0, 0.1]
    }

    if args.method not in ['adsn', 'adae', 'dsnw']:
        params_grid.pop('pretrain_num_epochs')

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param_dict in update_params_dict_list:
        main(args=args, update_params_dict=param_dict)
