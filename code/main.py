import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict

import data
import data_config
import train_dsn
import train_adae
import train_adsn
import train_coral
import fine_tuning
import ml_baseline


def generate_encoded_features(encoder, dataloader):
    """

    :param encoder:
    :param dataloader:
    :return:
    """
    encoder.eval()
    raw_feature_tensor = dataloader.dataset.tensors[0]
    label_tensor = dataloader.dataset.tensors[1]

    encoded_feature_tensor = encoder(raw_feature_tensor)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='adsn', choices=['adsn', 'dsn', 'adae', 'coral'])
    parser.add_argument('--drug', dest='drug', nargs='?', default='gem', choices=['gem', 'fu'])
    parser.add_argument('--thres', dest='auc_thres', nargs='?', default=0.8)
    parser.add_argument('--n', dest='n', nargs='?', default=1)

    args = parser.parse_args()

    if args.method == 'dsn':
        train_fn = train_dsn.train_dsn
    elif args.method == 'adae':
        train_fn = train_adae.train_adae
    elif args.method == 'coral':
        train_fn = train_coral.train_coral
    else:
        train_fn = train_adsn.train_adsn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)

    with open(os.path.join('model_save', args.method, 'train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            'model_save_folder': os.path.join('model_save', args.method)
        })

    ml_baseline_history = defaultdict(list)
    model_evaluation_history = defaultdict(list)

    random.seed(2020)
    seeds = random.sample(range(100000), k=int(args.n))

    for seed in seeds:
        s_dataloaders, t_dataloaders = data.get_unlabeled_dataloaders(
            gex_features_df=gex_features_df,
            seed=seed,
            batch_size=training_params['unlabeled']['batch_size']
        )

        labeled_ccle_dataloader, labeled_tcga_dataloader = data.get_labeled_dataloaders(
            gex_features_df=gex_features_df,
            seed=seed,
            batch_size=training_params['labeled']['batch_size'],
            drug=args.drug,
            auc_threshold=args.auc_thres
        )
        #start unlabeled training
        encoder, historys = train_fn(s_dataloaders=s_dataloaders,
                                     t_dataloaders=t_dataloaders,
                                     **wrap_training_params(training_params, type='unlabeled'))

        with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history_{seed}.pickle'), 'wb') as f:
            for history in historys:
                pickle.dump(dict(history), f)

        # generate encoded features
        ccle_encoded_feature_tensor, ccle_label_tensor = generate_encoded_features(encoder, labeled_ccle_dataloader)
        tcga_encoded_feature_tensor, tcga_label_tensor = generate_encoded_features(encoder, labeled_tcga_dataloader)
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
            **wrap_training_params(training_params, type='labeled')
        )

        with open(os.path.join(training_params['model_save_folder'], f'ft_train_history_{seed}.pickle'), 'wb') as f:
            for history in ft_historys:
                pickle.dump(dict(history), f)

    with open(os.path.join(training_params['model_save_folder'], 'ml_baseline_results.json'), 'w') as f:
        json.dump(ml_baseline_history, f)



