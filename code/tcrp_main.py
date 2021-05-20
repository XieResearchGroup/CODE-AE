from evaluation_utils import evaluate_target_classification_epoch, model_save_check
from itertools import chain
from mlp import MLP
from ae import AE
from encoder_decoder import EncoderDecoder
import os
import torch
import torch.nn as nn
import pandas as pd
import torch
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def tcrp_classification_train_step(model, dataloaders, loss_fn, device, optimizer, history, lr=1e-4, scheduler=None, clip=None):
    model.zero_grad()
    model.train()
    loss_q = 0

    model_weights = list(model.parameters())
    for dataloader in dataloaders:
        for param_cur, param_ori in zip(model.parameters(), model_weights):
            param_cur.data = param_ori.data
        batch = next(iter(dataloader))
        batch_size = len(batch[0])
        spt_x = batch[0][:batch_size // 2].to(device)
        spt_y = batch[1][:batch_size // 2].to(device)

        qry_x = batch[0][batch_size // 2:].to(device)
        qry_y = batch[1][batch_size // 2:].to(device)

        loss_s = loss_fn(model(spt_x), spt_y.double().unsqueeze(1))
        grad_s = torch.autograd.grad(loss_s, model.parameters())
        fast_weights = list(map(lambda p: p[1] - lr * p[0], zip(grad_s, model.parameters())))

        for param_cur, param_spt in zip(model.parameters(), fast_weights):
            param_cur.data = param_spt.data

        loss_q += loss_fn(model(qry_x), qry_y.double().unsqueeze(1))

    loss_q = loss_q / len(dataloaders)

    optimizer.zero_grad()
    loss_q.backward()

    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['bce'].append(loss_q.cpu().detach().item())

    return history


def fine_tune_encoder(train_dataloader, val_dataloader, tissue_dataloader_dict, seed, task_save_folder,
                      test_dataloader=None,
                      metric_name='auroc',
                      normalize_flag=False, **kwargs):
    autoencoder = AE(input_dim=kwargs['input_dim'],
                     latent_dim=kwargs['latent_dim'],
                     hidden_dims=kwargs['encoder_hidden_dims'],
                     noise_flag=False,
                     dop=kwargs['dop']).to(kwargs['device'])

    encoder = autoencoder.encoder
    target_decoder = MLP(input_dim=kwargs['latent_dim'],
                         output_dim=1,
                         hidden_dims=kwargs['classifier_hidden_dims']).to(kwargs['device'])
    target_classifier = EncoderDecoder(encoder=encoder, decoder=target_decoder, normalize_flag=normalize_flag).to(
        kwargs['device'])
    classification_loss = nn.BCEWithLogitsLoss()

    target_classification_train_history = defaultdict(list)
    target_classification_eval_train_history = defaultdict(list)
    target_classification_eval_val_history = defaultdict(list)
    target_classification_eval_test_history = defaultdict(list)

    lr = kwargs['lr']

    target_classification_params = [target_classifier.parameters()]
    target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params),
                                                        lr=lr)

    set_seed(2020)

    for epoch in range(kwargs['train_num_epochs']):
        if epoch % 50 == 0:
            print(f'Fine tuning epoch {epoch}')
        try:
            selected_tissues = random.sample(list(tissue_dataloader_dict), k=12)
        except ValueError:
            print('No enough tissues available')
            return None, (target_classification_train_history, target_classification_eval_train_history,
                               target_classification_eval_val_history, target_classification_eval_test_history)

        train_dataloaders = [tissue_dataloader_dict[tissue] for tissue in selected_tissues]
        target_classification_train_history = tcrp_classification_train_step(model=target_classifier,
                                                                             dataloaders=train_dataloaders,
                                                                             loss_fn=classification_loss,
                                                                             device=kwargs['device'],
                                                                             optimizer=target_classification_optimizer,
                                                                             history=target_classification_train_history)

        target_classification_eval_train_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                        dataloader=train_dataloader,
                                                                                        device=kwargs['device'],
                                                                                        history=target_classification_eval_train_history)
        target_classification_eval_val_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                      dataloader=val_dataloader,
                                                                                      device=kwargs['device'],
                                                                                      history=target_classification_eval_val_history)

        if test_dataloader is not None:
            target_classification_eval_test_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                           dataloader=test_dataloader,
                                                                                           device=kwargs['device'],
                                                                                           history=target_classification_eval_test_history)
        save_flag, stop_flag = model_save_check(history=target_classification_eval_val_history,
                                                metric_name=metric_name,
                                                tolerance_count=100)
        if save_flag:
            torch.save(target_classifier.state_dict(),
                       os.path.join(task_save_folder, f'target_classifier_{seed}.pt'))
        if stop_flag:
            break

    target_classifier.load_state_dict(
        torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))

    return target_classifier, (target_classification_train_history, target_classification_eval_train_history,
                               target_classification_eval_val_history, target_classification_eval_test_history)


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


def main(args, drug, update_params_dict):
    normalize_flag = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)

    with open(os.path.join('train_params.json'), 'r') as f:
        training_params = json.load(f)

    param_str = dict_to_str(update_params_dict)

    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            'model_save_folder': os.path.join('model_save', 'tcrp'),
            'es_flag': False
        })
    if args.pdtc_flag:
        task_save_folder = os.path.join('model_save', 'tcrp', args.measurement, 'pdtc', drug)
    else:
        task_save_folder = os.path.join('model_save', 'tcrp', args.measurement, drug)

    safe_make_dir(training_params['model_save_folder'])
    safe_make_dir(task_save_folder)

    random.seed(2020)
    ft_evaluation_metrics = defaultdict(list)
    labeled_dataloader_generator = data.get_labeled_tissue_dataloader_generator(
        gex_features_df=gex_features_df,
        drug=drug,
        seed=2020,
        batch_size=training_params['labeled']['batch_size'],
        ccle_measurement=args.measurement,
        threshold=args.a_thres,
        days_threshold=args.days_thres,
        pdtc_flag=args.pdtc_flag,
        n_splits=args.n)
    fold_count = 0
    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader, tissue_dataloader_dict in labeled_dataloader_generator:
        target_classifier, ft_historys = fine_tune_encoder(
            train_dataloader=train_labeled_ccle_dataloader,
            val_dataloader=test_labeled_ccle_dataloader,
            test_dataloader=labeled_tcga_dataloader,
            tissue_dataloader_dict=tissue_dataloader_dict,
            seed=fold_count,
            normalize_flag=normalize_flag,
            metric_name=args.metric,
            task_save_folder=task_save_folder,
            **wrap_training_params(training_params, type='labeled')
        )
        if target_classifier is None:
            break
        for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
            ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])
        fold_count += 1

    with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(ft_evaluation_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
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

    params_grid = {
        "train_num_epochs": [20000],
        "dop": [0.0, 0.1]
    }

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if args.pdtc_flag:
        drug_list = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0).index.tolist()
    else:
        drug_list = ['tgem', 'tfu', 'tem', 'gem', 'cis', 'sor', 'fu']

    for drug in drug_list:
        for param_dict in update_params_dict_list:
            main(args=args, drug=drug, update_params_dict=param_dict)
