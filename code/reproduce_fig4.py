import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import numpy as np
import data
import data_config
import train_code_adv
import fine_tuning



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




def main(args, drug):
    print('current config is {}'.format(args))
    print('current drug is {}'.format(drug))
    #update_params_dict=update_params_dict_list[0]


    train_fn = train_code_adv.train_code_adv


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)

    with open(os.path.join('model_save/train_params.json'), 'r') as f:
        training_params = json.load(f)

    # training_params['unlabeled'].update(update_params_dict)
    # param_str = dict_to_str(update_params_dict)

    if not args.norm_flag:
        method_save_folder = os.path.join('model_save', args.method)
        #  method_save_folder = os.path.join('model_save', f'{args.method}_es')
    else:
        method_save_folder = os.path.join('model_save' ,f'{args.method}_norm')
        #method_save_folder = os.path.join('model_save', f'{args.method}_norm_es')
    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            #'model_save_folder': os.path.join(method_save_folder, param_str),
            'es_flag': False,
            'retrain_flag': args.retrain_flag,
            'norm_flag': args.norm_flag
        })
    if args.pdtc_flag:
        task_save_folder = os.path.join(f'{method_save_folder}', args.measurement, 'pdtc', drug)
    else:
        task_save_folder = os.path.join(f'{method_save_folder}', args.measurement, drug)
    
    safe_make_dir(task_save_folder)
    random.seed(2020)

    print('start eval.... ')
    ft_evaluation_metrics = defaultdict(list)
    labeled_dataloader_generator = data.get_labeled_dataloader_generator(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=training_params['labeled']['batch_size'],
        drug=drug,
        ccle_measurement=args.measurement,
        threshold=args.a_thres,
        days_threshold=args.days_thres,
        pdtc_flag=args.pdtc_flag,
        n_splits=args.n)
    fold_count = 0
   
    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader in labeled_dataloader_generator:
       
        print(train_labeled_ccle_dataloader.dataset.tensors[1].sum())
        print(test_labeled_ccle_dataloader.dataset.tensors[1].sum())
        print(labeled_tcga_dataloader.dataset.tensors[1].sum())
      
        ft_historys = fine_tuning.reproduce_result(
            train_dataloader=train_labeled_ccle_dataloader,
            val_dataloader=test_labeled_ccle_dataloader,
            test_dataloader=labeled_tcga_dataloader,
            seed=fold_count,
            normalize_flag=args.norm_flag,
            metric_name=args.metric,
            task_save_folder=task_save_folder,
            **wrap_training_params(training_params, type='labeled')
        )
        
        ft_evaluation_metrics[args.metric].append(ft_historys[args.metric][0])
                
        fold_count += 1
        
    print('the final metrics are {}'.format(ft_evaluation_metrics) )
    final_report[drug].append(ft_evaluation_metrics[args.metric])
    with open(f'../result/CODEAE_Fig4_{args.metric}', 'w') as f:
        json.dump(final_report, f)     



if __name__ == '__main__':
    parser = argparse.ArgumentParser('CODE-AE reproducing result')
    parser.add_argument('--method', dest='method', nargs='?', default='code_adv',
                        choices=['code_adv', 'dsn', 'dsna','code_base', 'code_mmd', 'adae', 'coral', 'dae', 'vae', 'ae'])
    parser.add_argument('--metric', dest='metric', nargs='?', default='auprc', choices=['auroc', 'auprc'])

    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--a_thres', dest='a_thres', nargs='?', type=float, default=None)
    parser.add_argument('--d_thres', dest='days_thres', nargs='?', type=float, default=None)

    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=False)

    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()

    for args.metric in ['auroc', 'auprc']:
        final_report = defaultdict(list)
        for drug in ['tgem', 'tfu', 'tem', 'gem', 'cis', 'sor', 'fu']:
            main(args=args, drug=drug)