import gzip
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import data_config
from data_preprocessing import align_feature


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_unlabeled_dataloaders(gex_features_df, seed, batch_size):
    """
    CCLE as source domain, thus s_dataloaders
    Xena(TCGA) as target domain, thus t_dataloaders
    :param gex_features_df:
    :param seed:
    :param batch_size:
    :return:
    """
    set_seed(seed)
    ccle_sample_info_df = pd.read_csv(data_config.ccle_sample_file, index_col=0)
    with gzip.open(data_config.xena_sample_file) as f:
        xena_sample_info_df = pd.read_csv(f, sep='\t', index_col=0)
    xena_samples = xena_sample_info_df.index.intersection(gex_features_df.index)
    ccle_samples = gex_features_df.index.difference(xena_samples)
    xena_sample_info_df = xena_sample_info_df.loc[xena_samples]
    ccle_sample_info_df = ccle_sample_info_df.loc[ccle_samples.intersection(ccle_sample_info_df.index)]

    xena_df = gex_features_df.loc[xena_samples]
    ccle_df = gex_features_df.loc[ccle_samples]

    excluded_ccle_samples = []
    excluded_ccle_samples.extend(ccle_df.index.difference(ccle_sample_info_df.index))
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < 2].index
    excluded_ccle_samples.extend(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)

    to_split_ccle_df = ccle_df[~ccle_df.index.isin(excluded_ccle_samples)]
    train_ccle_df, test_ccle_df = train_test_split(to_split_ccle_df, test_size=0.1,
                                                   stratify=ccle_sample_info_df.loc[
                                                       to_split_ccle_df.index].primary_disease)
    test_ccle_df = test_ccle_df.append(ccle_df.loc[excluded_ccle_samples])
    train_xena_df, test_xena_df = train_test_split(xena_df, test_size=len(test_ccle_df) / len(xena_df),
                                                   stratify=xena_sample_info_df['_primary_disease'])

    xena_dataset = TensorDataset(
        torch.from_numpy(xena_df.values.astype('float32'))
    )

    ccle_dataset = TensorDataset(
        torch.from_numpy(ccle_df.value.astype('float32'))
    )

    train_xena_dateset = TensorDataset(
        torch.from_numpy(train_xena_df.values.astype('float32')))
    test_xena_dateset = TensorDataset(
        torch.from_numpy(test_xena_df.values.astype('float32')))
    train_ccle_dateset = TensorDataset(
        torch.from_numpy(train_ccle_df.values.astype('float32')))
    test_ccle_dateset = TensorDataset(
        torch.from_numpy(test_ccle_df.values.astype('float32')))

    xena_dataloader = DataLoader(xena_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    train_xena_dataloader = DataLoader(train_xena_dateset,
                                       batch_size=batch_size,
                                       shuffle=True)
    test_xena_dataloader = DataLoader(test_xena_dateset,
                                      batch_size=batch_size,
                                      shuffle=True)

    ccle_data_loader = DataLoader(ccle_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True
                                  )

    train_ccle_dataloader = DataLoader(train_ccle_dateset,
                                       batch_size=batch_size,
                                       shuffle=True, drop_last=True)
    test_ccle_dataloader = DataLoader(test_ccle_dateset,
                                      batch_size=batch_size,
                                      shuffle=True)
    return (ccle_data_loader, test_ccle_dataloader), (xena_dataloader, test_xena_dataloader)
    # return (train_ccle_dataloader, test_ccle_dataloader), (train_xena_dataloader, test_xena_dataloader)


def get_labeled_dataloaders(gex_features_df, seed, batch_size, ft_flag=False, drug='gem', auc_threshold=0.80):
    """

    :param gex_features_df:
    :param ft_flag:
    :param seed:
    :param batch_size:
    :param drug:
    :param auc_threshold:
    :return:
    """
    if drug == 'gem':
        drugs_to_keep = ['Gemcitabine']
    else:
        drugs_to_keep = ['5-Fluorouracil']

    non_feature_file_path = os.path.join(data_config.preprocessed_data_folder, f'{drug}_non_gex.csv')
    res_feature_file_path = os.path.join(data_config.preprocessed_data_folder, f'{drug}_res_gex.csv')

    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)

    non_feature_df = pd.read_csv(non_feature_file_path, index_col=0)
    _, non_feature_df = align_feature(gex_features_df, non_feature_df)

    res_feature_df = pd.read_csv(res_feature_file_path, index_col=0)
    _, res_feature_df = align_feature(gex_features_df, res_feature_df)

    raw_tcga_feature_df = pd.concat([non_feature_df, res_feature_df])

    tcga_label = np.ones(raw_tcga_feature_df.shape[0], dtype='int32')
    tcga_label[:len(non_feature_df)] = 0
    tcga_label_df = pd.DataFrame(tcga_label, index=raw_tcga_feature_df.index, columns=['label'])

    gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', 'AUC']]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', 'AUC']]
    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values='AUC', index='COSMIC_ID', columns='DRUG_NAME')
    ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    gdsc_sample_info = pd.read_csv(data_config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]

    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']

    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]

    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < auc_threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    if ft_flag:
        train_labeled_ccle_df, test_labeled_ccle_df, train_ccle_labels, test_ccle_labels = train_test_split(
            ccle_labeled_feature_df.values,
            ccle_labels.values,
            test_size=0.1,
            stratify=ccle_labels.values)

        train_labeled_ccle_dateset = TensorDataset(
            torch.from_numpy(train_labeled_ccle_df.astype('float32')),
            torch.from_numpy(train_ccle_labels))
        test_labeled_ccle_df = TensorDataset(
            torch.from_numpy(test_labeled_ccle_df.astype('float32')),
            torch.from_numpy(test_ccle_labels))

        train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_df,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    labeled_ccle_dateset = TensorDataset(
        torch.from_numpy(ccle_labeled_feature_df.values.astype('float32')),
        torch.from_numpy(ccle_labels.values))
    labeled_tcga_dateset = TensorDataset(
        torch.from_numpy(raw_tcga_feature_df.values.astype('float32')),
        torch.from_numpy(tcga_label_df.values))

    labeled_ccle_dataloader = DataLoader(labeled_ccle_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return (train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader) if ft_flag else (
        labeled_ccle_dataloader, labeled_tcga_dataloader)


def get_adae_unlabeled_dataloaders(gex_features_df, seed, batch_size, pos_gender='female'):
    sex_label_df = pd.read_table(data_config.adae_sex_label_file, index_col=0)
    pos_samples = gex_features_df.index.intersection(sex_label_df.index[sex_label_df.iloc[:, 0] == pos_gender])
    neg_samples = gex_features_df.index.intersection(sex_label_df.index[sex_label_df.iloc[:, 0] != pos_gender])

    s_df = gex_features_df.loc[pos_samples]
    t_df = gex_features_df.loc[neg_samples]

    s_dataset = TensorDataset(
        torch.from_numpy(s_df.values.astype('float32'))
    )

    t_dataset = TensorDataset(
        torch.from_numpy(t_df.values.astype('float32'))
    )

    s_dataloader = DataLoader(s_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True
                              )

    t_dataloader = DataLoader(t_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True
                              )

    return s_dataloader, t_dataloader


def get_adae_labeled_dataloaders(gex_features_df, seed, batch_size, pos_gender='female', ft_flag=False):
    """

    :param gex_features_df:
    :param seed:
    :param batch_size:
    :param pos_gender:
    :return:
    """
    sex_label_df = pd.read_table(data_config.adae_sex_label_file, index_col=0)
    subtype_label_df = pd.read_table(data_config.adae_subtype_label_file, index_col=0)
    gex_features_df = gex_features_df.loc[sex_label_df.index.intersection(gex_features_df.index)]
    subtype_label_df = subtype_label_df.loc[gex_features_df.index]
    assert all(gex_features_df.index == subtype_label_df.index)

    train_samples = gex_features_df.index.intersection(sex_label_df.index[sex_label_df.iloc[:, 0] == pos_gender])
    test_samples = gex_features_df.index.intersection(sex_label_df.index[sex_label_df.iloc[:, 0] != pos_gender])

    train_df = gex_features_df.loc[train_samples]
    test_df = gex_features_df.loc[test_samples]

    if ft_flag:
        train_df, val_df, train_labels, val_labels = train_test_split(
            train_df.values,
            subtype_label_df.loc[train_samples].values,
            test_size=0.1,
            stratify=subtype_label_df.loc[train_samples].values)

        train_labeled_dataset = TensorDataset(
            torch.from_numpy(train_df.values.astype('float32')),
            torch.from_numpy(train_labels)
        )

        val_labeled_dataset = TensorDataset(
            torch.from_numpy(val_df.values.astype('float32')),
            torch.from_numpy(val_labels)
        )

        val_labeled_dataloader = DataLoader(val_labeled_dataset,
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

    else:
        train_labeled_dataset = TensorDataset(
            torch.from_numpy(train_df.values.astype('float32')),
            torch.from_numpy(subtype_label_df.loc[train_samples].values)
        )

    test_labeled_dataset = TensorDataset(
        torch.from_numpy(test_df.values.astype('float32')),
        torch.from_numpy(subtype_label_df.loc[test_samples].values)
    )

    train_labeled_dataloader = DataLoader(train_labeled_dataset,
                                          batch_size=batch_size,
                                          shuffle=True
                                          )

    test_labeled_dataloader = DataLoader(test_labeled_dataset,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )

    return (train_labeled_dataloader, val_labeled_dataloader,
            test_labeled_dataloader) if ft_flag else train_labeled_dataloader, test_labeled_dataloader
