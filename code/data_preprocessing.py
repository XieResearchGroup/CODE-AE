import numpy as np
import pandas as pd
import os
import re
from sklearn.cluster import KMeans

import data_config


def align_feature(df1, df2):
    """
    match/align two data frames' columns (features)
    :param df1:
    :param df2:
    :return:
    """
    matched_features = list(set(df1.columns.tolist()) & set(df2.columns.tolist()))
    matched_features.sort()
    print('Aligned dataframes have {} features in common'.format(len(matched_features)))
    return df1[matched_features], df2[matched_features]


def generate_kmeans_features(df, k=1000):
    """
    Produce features sets based on kmeans++ over original dataset
    :param df: pandas.DataFrame, (sample, features)
    :param k:
    :return:
    """
    kmeans = KMeans(n_clusters=1000, random_state=2020).fit(df.transpose())
    cluster_centers = kmeans.cluster_centers_
    encoded_df = pd.DataFrame(cluster_centers.T, index=df.index)

    return encoded_df


def filter_features(df, mean_tres=1.0, std_thres=0.5):
    """
    filter genes of low information burden
    mean: sparsity threshold, std: variability threshold
    :param df: samples X features
    :param mean_tres:
    :param std_thres:
    :return:
    """
    df = df.loc[:, df.apply(lambda col: col.isna().sum()) == 0]
    feature_stds = df.std()
    feature_means = df.mean()
    std_to_drop = feature_stds[list(np.where(feature_stds <= std_thres)[0])].index.tolist()
    mean_to_drop = feature_means[list(np.where(feature_means <= mean_tres)[0])].index.tolist()
    to_drop = list(set(std_to_drop) | set(mean_to_drop))
    df.drop(labels=to_drop, axis=1, inplace=True)
    return df


def filter_with_MAD(df, k=5000):
    """
    pick top k MAD features
    :param df:
    :param k:
    :return:
    """
    result = df[(df - df.median()).abs().median().nlargest(k).index.tolist()]
    return result


def filter_features_with_list(df, feature_list):
    features_to_keep = list(set(df.columns.tolist()) & set(feature_list))
    return df[features_to_keep]


def preprocess_target_data(output_file_path=None):
    # keep only tcga classified samples
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

    gdsc_drug_sensitivity = pd.read_csv(data_config.gdsc_raw_target_file, index_col=0)

    gdsc_drug_sensitivity.drop(axis=1, columns=[gdsc_drug_sensitivity.columns[0]], inplace=True)
    gdsc_drug_sensitivity.index = gdsc_drug_sensitivity.index.map(gdsc_sample_mapping_dict)
    target_df = gdsc_drug_sensitivity.loc[gdsc_drug_sensitivity.index.dropna()]
    target_df = target_df.astype('float32')
    if output_file_path:
        target_df.to_csv(output_file_path + '.csv', index_label='Sample')
    return target_df


def get_tcga_first_treatment_df():
    drug_mapping_dict = pd.read_csv(data_config.tcga_drug_name_mapping_file, index_col=0).to_dict()['Correction']
    pancancer_drug_history_df = None
    for cancer_type in os.listdir(data_config.tcga_clinical_folder):
        sub_folder = os.path.join(data_config.tcga_clinical_folder, cancer_type)
        if os.path.isdir(sub_folder):
            drug_history_pattern = re.compile(f'nationwidechildrens.org_clinical_drug_{cancer_type.lower()}.txt')
            history_columns_to_keep = ['bcr_patient_barcode', 'pharmaceutical_therapy_drug_name',
                                       'pharmaceutical_tx_started_days_to']
            for file in os.listdir(sub_folder):
                if re.match(drug_history_pattern, file):
                    print(f'{cancer_type}: {file}')
                    drug_history_df = pd.read_csv(os.path.join(sub_folder, file), sep='\t')
                    drug_history_df = drug_history_df.drop(drug_history_df.index[:2])[history_columns_to_keep]
                    drug_history_df['pharmaceutical_therapy_drug_name'] = drug_history_df[
                        'pharmaceutical_therapy_drug_name'].map(drug_mapping_dict)
                    drug_history_df.dropna(inplace=True)
                    drug_history_df['tcga_project'] = cancer_type
                    pancancer_drug_history_df = pd.concat([pancancer_drug_history_df, drug_history_df])

    pancancer_drug_history_df = pancancer_drug_history_df.loc[
        pancancer_drug_history_df['pharmaceutical_tx_started_days_to'] != '[Not Available]']
    pancancer_drug_history_df = pancancer_drug_history_df.loc[
        pancancer_drug_history_df['pharmaceutical_tx_started_days_to'] != '[Discrepancy]']
    pancancer_drug_history_df = pancancer_drug_history_df.loc[
        pancancer_drug_history_df['pharmaceutical_tx_started_days_to'] != '[Completed]']
    pancancer_drug_history_df['pharmaceutical_tx_started_days_to'] = pancancer_drug_history_df[
        'pharmaceutical_tx_started_days_to'].astype('int')
    first_treatment_df = pancancer_drug_history_df.merge(pancancer_drug_history_df.groupby('bcr_patient_barcode')[
                                                             'pharmaceutical_tx_started_days_to'].min().reset_index())

    return first_treatment_df, pancancer_drug_history_df


def get_tcga_response_df():
    drug_mapping_dict = pd.read_csv(data_config.tcga_drug_name_mapping_file, index_col=0).to_dict()['Correction']
    pancancer_response_df = None
    # cols = set()
    for cancer_type in os.listdir(data_config.tcga_clinical_folder):
        sub_folder = os.path.join(data_config.tcga_clinical_folder, cancer_type)
        if os.path.isdir(sub_folder):
            drug_response_pattern1 = re.compile(f'nationwidechildrens.org_clinical.*_nte_{cancer_type.lower()}.txt')
            drug_response_pattern2 = re.compile(
                f'nationwidechildrens.org_clinical.*_follow_up.*_{cancer_type.lower()}.txt')

            for file in os.listdir(sub_folder):
                if re.match(drug_response_pattern1, file) or re.match(drug_response_pattern2, file):
                    drug_response_df = pd.read_csv(os.path.join(sub_folder, file), sep='\t')
                    # cols = cols.union(set(drug_response_df.columns.tolist()))
                    # print(drug_response_df.columns)
                    if 'days_to_new_tumor_event_after_initial_treatment' in drug_response_df.columns or 'new_tumor_event_dx_days_to' in drug_response_df.columns:
                        print(f'{cancer_type}: {file}')
                        history_columns_to_keep = ['bcr_patient_barcode',
                                                   'days_to_new_tumor_event_after_initial_treatment']
                        try:
                            drug_response_df = drug_response_df.drop(drug_response_df.index[:2])[
                                history_columns_to_keep]
                        except KeyError:
                            history_columns_to_keep = ['bcr_patient_barcode', 'new_tumor_event_dx_days_to']
                            drug_response_df = drug_response_df.drop(drug_response_df.index[:2])[
                                history_columns_to_keep]
                            drug_response_df['days_to_new_tumor_event_after_initial_treatment'] = drug_response_df[
                                'new_tumor_event_dx_days_to']
                            drug_response_df.drop(columns=['new_tumor_event_dx_days_to'], inplace=True)
                        drug_response_df.dropna(inplace=True)
                        drug_response_df['tcga_project'] = cancer_type
                        pancancer_response_df = pd.concat([pancancer_response_df, drug_response_df])

    pancancer_response_df = pancancer_response_df.loc[
        pancancer_response_df['days_to_new_tumor_event_after_initial_treatment'] != '[Not Available]']
    pancancer_response_df = pancancer_response_df.loc[
        pancancer_response_df['days_to_new_tumor_event_after_initial_treatment'] != '[Not Applicable]']
    pancancer_response_df = pancancer_response_df.loc[
        pancancer_response_df['days_to_new_tumor_event_after_initial_treatment'] != '[Discrepancy]']
    pancancer_response_df = pancancer_response_df.loc[
        pancancer_response_df['days_to_new_tumor_event_after_initial_treatment'] != '[Completed]']

    pancancer_response_df['days_to_new_tumor_event_after_initial_treatment'] = pancancer_response_df[
        'days_to_new_tumor_event_after_initial_treatment'].astype('int')
    first_response_df = pancancer_response_df.merge(pancancer_response_df.groupby('bcr_patient_barcode')[
                                                        'days_to_new_tumor_event_after_initial_treatment'].min().reset_index())

    # pprint(cols)
    return first_response_df, pancancer_response_df


if __name__ == '__main__':
    pass
