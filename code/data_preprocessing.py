import numpy as np
import pandas as pd
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

def generate_kmeans_features(df, k= 1000):
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
    #keep only tcga classified samples
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


if __name__ == '__main__':
    pass
