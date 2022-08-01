import gzip
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

import data_config
from data_preprocessing import align_feature


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_unlabeled_dataloaders(gex_features_df, seed, batch_size, ccle_only=False):
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
                                                   stratify=xena_sample_info_df['_primary_disease'],
                                                   random_state=seed)

    xena_dataset = TensorDataset(
        torch.from_numpy(xena_df.values.astype('float32'))
    )

    ccle_dataset = TensorDataset(
        torch.from_numpy(ccle_df.values.astype('float32'))
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
    if ccle_only:
        return (ccle_data_loader, test_ccle_dataloader), (ccle_data_loader, test_ccle_dataloader)
    else:
        return (ccle_data_loader, test_ccle_dataloader), (xena_dataloader, test_xena_dataloader)


def get_tcga_labeled_dataloaders(gex_features_df, drug, batch_size, days_threshold=None, tcga_cancer_type=None):
    if tcga_cancer_type is not None:
        raise NotImplementedError("Only support pan-cancer")

    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()

    tcga_treatment_df = pd.read_csv(data_config.tcga_first_treatment_file)
    tcga_response_df = pd.read_csv(data_config.tcga_first_response_file)

    tcga_treatment_df.drop_duplicates(subset=['bcr_patient_barcode'], keep=False, inplace=True)
    tcga_treatment_df.set_index('bcr_patient_barcode', inplace=True)

    tcga_response_df.drop_duplicates(inplace=True)
    tcga_response_df.drop_duplicates(subset=['bcr_patient_barcode'], inplace=True)
    tcga_response_df.set_index('bcr_patient_barcode', inplace=True)

    tcga_drug_barcodes = tcga_treatment_df.index[tcga_treatment_df['pharmaceutical_therapy_drug_name'] == drug]
    drug_tcga_response_df = tcga_response_df.loc[tcga_drug_barcodes.intersection(tcga_response_df.index)]
    labeled_tcga_gex_feature_df = tcga_gex_feature_df.loc[
        drug_tcga_response_df.index.intersection(tcga_gex_feature_df.index)]
    labeled_df = tcga_response_df.loc[labeled_tcga_gex_feature_df.index]
    # print(labeled_df)
    assert (all(labeled_df.index == labeled_tcga_gex_feature_df.index))

    if days_threshold is None:
        days_threshold = np.median(labeled_df.days_to_new_tumor_event_after_initial_treatment)

    drug_label = np.array(labeled_df.days_to_new_tumor_event_after_initial_treatment > days_threshold, dtype='int32')
    # drug_label = np.array(labeled_df.treatment_outcome_at_tcga_followup.apply(
    #     lambda s: s not in ['Progressive Disease', 'Stable Disease', 'Persistant Disease']), dtype='int32')

    # drug_label_df = pd.DataFrame(drug_label, index=labeled_df.index, columns=['label'])

    labeled_tcga_dateset = TensorDataset(
        torch.from_numpy(labeled_tcga_gex_feature_df.values.astype('float32')),
        torch.from_numpy(drug_label))

    labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return labeled_tcga_dataloader


def get_tcga_preprocessed_labeled_dataloaders(gex_features_df, drug, batch_size):
    if drug not in ['gem', 'fu']:
        raise NotImplementedError('Only support gem or fu!')
    non_feature_file_path = os.path.join(data_config.preprocessed_data_folder, f'{drug}_non_gex.csv')
    res_feature_file_path = os.path.join(data_config.preprocessed_data_folder, f'{drug}_res_gex.csv')

    non_feature_df = pd.read_csv(non_feature_file_path, index_col=0)
    _, non_feature_df = align_feature(gex_features_df, non_feature_df)

    res_feature_df = pd.read_csv(res_feature_file_path, index_col=0)
    _, res_feature_df = align_feature(gex_features_df, res_feature_df)

    raw_tcga_feature_df = pd.concat([non_feature_df, res_feature_df])

    tcga_label = np.ones(raw_tcga_feature_df.shape[0], dtype='int32')
    tcga_label[:len(non_feature_df)] = 0
    # tcga_label_df = pd.DataFrame(tcga_label, index=raw_tcga_feature_df.index, columns=['label'])

    labeled_tcga_dateset = TensorDataset(
        torch.from_numpy(raw_tcga_feature_df.values.astype('float32')),
        torch.from_numpy(tcga_label))

    labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return labeled_tcga_dataloader


def get_pdtc_labeled_dataloaders(drug, batch_size, threshold=None, measurement='AUC'):
    pdtc_features_df = pd.read_csv(data_config.pdtc_gex_file, index_col=0)
    target_df = pd.read_csv(data_config.pdtc_target_file, index_col=0, sep='\t')
    drug_target_df = target_df.loc[target_df.Drug == drug]
    labeled_samples = drug_target_df.index.intersection(pdtc_features_df.index)
    drug_target_vec = drug_target_df.loc[labeled_samples, measurement]
    drug_feature_df = pdtc_features_df.loc[labeled_samples]

    assert all(drug_target_vec.index == drug_target_vec.index)

    if threshold is None:
        threshold = np.median(drug_target_vec)

    drug_label_vec = (drug_target_vec < threshold).astype('int')

    labeled_pdtc_dateset = TensorDataset(
        torch.from_numpy(drug_feature_df.values.astype('float32')),
        torch.from_numpy(drug_label_vec.values))

    labeled_pdtc_dataloader = DataLoader(labeled_pdtc_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return labeled_pdtc_dataloader


def get_ccle_labeled_dataloaders(gex_features_df, seed, drug, batch_size, ft_flag=False, threshold=None,
                                 measurement='AUC'):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
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

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    if ft_flag:
        train_labeled_ccle_df, test_labeled_ccle_df, train_ccle_labels, test_ccle_labels = train_test_split(
            ccle_labeled_feature_df.values,
            ccle_labels.values,
            test_size=0.1,
            stratify=ccle_labels.values,
            random_state=seed
        )

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

    labeled_ccle_dataloader = DataLoader(labeled_ccle_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return (train_labeled_ccle_dataloader, test_labeled_ccle_dataloader) if ft_flag else labeled_ccle_dataloader


def get_ccle_labeled_dataloader_generator(gex_features_df, drug, batch_size, seed=2020, threshold=None,
                                          measurement='AUC', n_splits=5):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
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

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    s_kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]

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

        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader


def get_labeled_dataloaders(gex_features_df, drug, seed, batch_size, ccle_measurement='AUC', threshold=None,
                            days_threshold=None,
                            ft_flag=False,
                            pdtc_flag=False):
    """
    sensitive (responder): 1
    resistant (non-responder): 0
    """
    if pdtc_flag:
        drug_mapping_df = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0)
    else:
        drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    if drug in ['tgem', 'tfu']:
        gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']
    else:
        gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug, 'drug_name']

    print(f'Drug: {drug}, TCGA (PDTC): {drug_name}, GDSC: {gdsc_drug}')

    ccle_labeled_dataloaders = get_ccle_labeled_dataloaders(gex_features_df=gex_features_df,
                                                            threshold=threshold, seed=seed, drug=gdsc_drug,
                                                            batch_size=batch_size, ft_flag=ft_flag,
                                                            measurement=ccle_measurement)
    if pdtc_flag:
        test_labeled_dataloaders = get_pdtc_labeled_dataloaders(drug=drug_name,
                                                                batch_size=batch_size,
                                                                threshold=threshold,
                                                                measurement=ccle_measurement)
    else:
        if drug in ['tgem', 'tfu']:
            test_labeled_dataloaders = get_tcga_preprocessed_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                                 drug=drug[1:],
                                                                                 batch_size=batch_size)
        else:
            test_labeled_dataloaders = get_tcga_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                    drug=drug_name,
                                                                    batch_size=batch_size,
                                                                    days_threshold=days_threshold)

    return ccle_labeled_dataloaders, test_labeled_dataloaders


def get_labeled_dataloader_generator(gex_features_df, drug, seed, batch_size, ccle_measurement='AUC', threshold=None,
                                     days_threshold=None,
                                     pdtc_flag=False,
                                     n_splits=5):
    """
    sensitive (responder): 1
    resistant (non-responder): 0

    """
    if pdtc_flag:
        drug_mapping_df = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0)
    else:
        drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    if drug in ['tgem', 'tfu']:
        gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']
    else:
        gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug, 'drug_name']

    print(f'Drug: {drug}, TCGA (PDTC): {drug_name}, GDSC: {gdsc_drug}')

    ccle_labeled_dataloader_generator = get_ccle_labeled_dataloader_generator(gex_features_df=gex_features_df,
                                                                              seed=seed,
                                                                              drug=gdsc_drug,
                                                                              batch_size=batch_size,
                                                                              threshold=threshold,
                                                                              measurement=ccle_measurement,
                                                                              n_splits=n_splits)

    if pdtc_flag:
        test_labeled_dataloaders = get_pdtc_labeled_dataloaders(drug=drug_name,
                                                                batch_size=batch_size,
                                                                threshold=threshold,
                                                                measurement=ccle_measurement)
    else:
        if drug in ['tgem', 'tfu']:
            test_labeled_dataloaders = get_tcga_preprocessed_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                                 drug=drug[1:],
                                                                                 batch_size=batch_size)
        else:
            test_labeled_dataloaders = get_tcga_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                    drug=drug_name,
                                                                    batch_size=batch_size,
                                                                    days_threshold=days_threshold)

    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader in ccle_labeled_dataloader_generator:
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, test_labeled_dataloaders


def get_ccle_labeled_tissue_dataloader_generator(gex_features_df, drug, batch_size, seed=2020, threshold=None,
                                                 measurement='AUC', n_splits=5, num_samples=12):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
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

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    ccle_sample_info.set_index('DepMap_ID', inplace=True)
    ccle_label_tissues = ccle_sample_info.loc[ccle_labeled_samples, 'lineage']
    assert all(ccle_labels.index == ccle_label_tissues.index)

    s_kfold = StratifiedKFold(n_splits=n_splits, random_state=seed)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]

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

        train_samples = ccle_labeled_samples[train_index]
        train_sample_info = ccle_sample_info.loc[train_samples]
        train_tissue_counts = train_sample_info.lineage.value_counts()
        valid_tissues = train_tissue_counts.index[train_tissue_counts >= num_samples].tolist()
        tissue_dataloader_dict = dict()

        for tissue in valid_tissues:
            tissue_samples = train_sample_info.index[train_sample_info.lineage == tissue]
            tissue_feature_df = ccle_labeled_feature_df.loc[tissue_samples]
            tissue_label_df = ccle_labels.loc[tissue_samples]
            tissue_labeled_ccle_dateset = TensorDataset(
                torch.from_numpy(tissue_feature_df.values.astype('float32')),
                torch.from_numpy(tissue_label_df.values))

            tissue_labeled_ccle_dataloader = DataLoader(tissue_labeled_ccle_dateset,
                                                        batch_size=num_samples,
                                                        shuffle=True,
                                                        drop_last=True)
            tissue_dataloader_dict[tissue] = tissue_labeled_ccle_dataloader

        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, tissue_dataloader_dict


def get_labeled_tissue_dataloader_generator(gex_features_df, drug, seed, batch_size, ccle_measurement='AUC',
                                            threshold=None,
                                            days_threshold=None,
                                            pdtc_flag=False,
                                            n_splits=5):
    """
    sensitive (responder): 1
    resistant (non-responder): 0

    """
    if pdtc_flag:
        drug_mapping_df = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0)
    else:
        drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    if drug in ['tgem', 'tfu']:
        gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']
    else:
        gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug, 'drug_name']

    print(f'Drug: {drug}, TCGA (PDTC): {drug_name}, GDSC: {gdsc_drug}')

    ccle_labeled_dataloader_generator = get_ccle_labeled_tissue_dataloader_generator(gex_features_df=gex_features_df,
                                                                                     seed=seed,
                                                                                     drug=gdsc_drug,
                                                                                     batch_size=batch_size,
                                                                                     threshold=threshold,
                                                                                     measurement=ccle_measurement,
                                                                                     n_splits=n_splits)

    if pdtc_flag:
        test_labeled_dataloaders = get_pdtc_labeled_dataloaders(drug=drug_name,
                                                                batch_size=batch_size,
                                                                threshold=threshold,
                                                                measurement=ccle_measurement)
    else:
        if drug in ['tgem', 'tfu']:
            test_labeled_dataloaders = get_tcga_preprocessed_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                                 drug=drug[1:],
                                                                                 batch_size=batch_size)
        else:
            test_labeled_dataloaders = get_tcga_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                    drug=drug_name,
                                                                    batch_size=batch_size,
                                                                    days_threshold=days_threshold)

    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, tissue_dataloader_dict in ccle_labeled_dataloader_generator:
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, test_labeled_dataloaders, tissue_dataloader_dict


def get_adae_unlabeled_dataloaders(gex_features_df, batch_size, pos_gender='female'):
    sex_label_df = pd.read_csv(data_config.adae_sex_label_file, index_col=0, sep='\t')
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

    return (s_dataloader, s_dataloader), (t_dataloader, t_dataloader)


def get_adae_labeled_dataloaders(gex_features_df, seed, batch_size, pos_gender='female', ft_flag=False):
    """

    :param gex_features_df:
    :param seed:
    :param batch_size:
    :param pos_gender:
    :return:
    """
    sex_label_df = pd.read_csv(data_config.adae_sex_label_file, index_col=0, sep='\t')
    subtype_label_df = pd.read_csv(data_config.adae_subtype_label_file, index_col=0, sep='\t')
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
            stratify=subtype_label_df.loc[train_samples].values,
            random_state=seed)

        train_labeled_dataset = TensorDataset(
            torch.from_numpy(train_df.astype('float32')),
            torch.from_numpy(train_labels.ravel())
        )

        val_labeled_dataset = TensorDataset(
            torch.from_numpy(val_df.astype('float32')),
            torch.from_numpy(val_labels.ravel())
        )

        val_labeled_dataloader = DataLoader(val_labeled_dataset,
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

    else:
        train_labeled_dataset = TensorDataset(
            torch.from_numpy(train_df.values.astype('float32')),
            torch.from_numpy(subtype_label_df.loc[train_samples].values.ravel())
        )

    test_labeled_dataset = TensorDataset(
        torch.from_numpy(test_df.values.astype('float32')),
        torch.from_numpy(subtype_label_df.loc[test_samples].values.ravel())
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
            test_labeled_dataloader) if ft_flag else (train_labeled_dataloader, test_labeled_dataloader)
