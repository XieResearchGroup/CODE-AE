import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import itertools

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


    normalize_flag = 'dsn' in args.method

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)

    with open(os.path.join('model_save', args.method, 'train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            'model_save_folder': os.path.join('model_save', args.method, args.drug),
            'es_flag': False
        })

    training_params['unlabeled'].update(update_params_dict)

    param_str = dict_to_str(update_params_dict)
    safe_make_dir(training_params['model_save_folder'])

    ml_baseline_history = defaultdict(list)
    model_evaluation_history = defaultdict(list)

    s_dataloaders, t_dataloaders = data.get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=training_params['unlabeled']['batch_size']
    )

    labeled_ccle_dataloader, labeled_tcga_dataloader = data.get_labeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=training_params['labeled']['batch_size'],
        drug=args.drug,
        auc_threshold=args.auc_thres
    )

    # start unlabeled training
    encoder, historys = train_fn(s_dataloaders=s_dataloaders,
                                 t_dataloaders=t_dataloaders,
                                 **wrap_training_params(training_params, type='unlabeled'))

    with open(os.path.join(training_params['model_save_folder'], f'{param_str}_unlabel_train_history.pickle'), 'wb') as f:
        for history in historys:
            pickle.dump(dict(history), f)

    # generate encoded features
    ccle_encoded_feature_tensor, ccle_label_tensor = generate_encoded_features(encoder, labeled_ccle_dataloader,
                                                                               normalize_flag=normalize_flag)
    tcga_encoded_feature_tensor, tcga_label_tensor = generate_encoded_features(encoder, labeled_tcga_dataloader,
                                                                               normalize_flag=normalize_flag)
    # build baseline ml models for encoded features
    ml_baseline_history['rf'].append(
        ml_baseline.n_time_cv(
            model_fn=ml_baseline.classify_with_rf,
            n=1,
            train_data=(
                ccle_encoded_feature_tensor.detach().cpu().numpy(),
                ccle_label_tensor.detach().cpu().numpy()
            ),
            test_data=(
                tcga_encoded_feature_tensor.detach().cpu().numpy(),
                tcga_label_tensor.detach().cpu().numpy()
            )
        )[1]
    )

    ml_baseline_history['enet'].append(
        ml_baseline.n_time_cv(
            model_fn=ml_baseline.classify_with_enet,
            n=1,
            train_data=(
                ccle_encoded_feature_tensor.detach().cpu().numpy(),
                ccle_label_tensor.detach().cpu().numpy()
            ),
            test_data=(
                tcga_encoded_feature_tensor.detach().cpu().numpy(),
                tcga_label_tensor.detach().cpu().numpy()
            )
        )[1]
    )

    # start fine-tuning encoder
    target_classifier, ft_historys = fine_tuning.fine_tune_encoder(
        encoder=encoder,
        train_dataloader=labeled_ccle_dataloader,
        val_dataloader=labeled_tcga_dataloader,
        test_dataloader=labeled_tcga_dataloader,
        normalize_flag=normalize_flag,
        **wrap_training_params(training_params, type='labeled')
    )

    with open(os.path.join(training_params['model_save_folder'], f'{param_str}_ft_train_history.pickle'), 'wb') as f:
        for history in ft_historys:
            pickle.dump(dict(history), f)

    with open(os.path.join(training_params['model_save_folder'], f'{param_str}_ml_baseline_results.json'), 'w') as f:
        json.dump(ml_baseline_history, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='adsn',
                        choices=['adsn', 'dsn', 'ndsn', 'mdsn', 'dsnw', 'adae', 'coral', 'dae', 'vae', 'ae'])
    parser.add_argument('--drug', dest='drug', nargs='?', default='gem', choices=['gem', 'fu'])
    parser.add_argument('--thres', dest='auc_thres', nargs='?', default=0.8)
    parser.add_argument('--n', dest='n', nargs='?', default=1)

    args = parser.parse_args()

    params_grid = {
        #"pretrain_num_epochs": [0, 10, 30, 50, 100, 150, 200, 250, 300],
        "pretrain_num_epochs": [0],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000]
    }

    if args.method not in ['adsn', 'adae', 'dsnw']:
        params_grid.pop('pretrain_num_epochs')

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param_dict in update_params_dict_list:
        main(args=args, update_params_dict=param_dict)







