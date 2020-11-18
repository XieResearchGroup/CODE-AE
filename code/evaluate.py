import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score


def drug_wise_evaluation(truth_df, pred_df):
    """
    classification performance evaluation
    :param truth_df:
    :param pred_df:
    :return:
    """
    pred_df = pred_df.loc[truth_df.index]
    per_drug_measurement = defaultdict(dict)
    for drug in truth_df.columns:
        samples = sorted(truth_df.index[~truth_df[drug].isna()])
        pred_samples = sorted(pred_df.index[pred_df[drug] != -1])
        if not samples == pred_samples:
            print(f'{drug}: invalid prediction!')
            pass
        else:
            truth_vec = truth_df.loc[samples, drug]
            pred_vec = pred_df.loc[samples, drug]
            per_drug_measurement['auroc'][drug] = roc_auc_score(y_true=truth_vec, y_score=pred_vec)
            per_drug_measurement['postive_rate'][drug] = truth_vec.sum() / len(samples)
            per_drug_measurement['ap'][drug] = average_precision_score(y_true=truth_vec, y_score=pred_vec)

    return pd.DataFrame.from_dict(per_drug_measurement)

def cell_wise_evaluation(truth_df, pred_df):
    """
    classification performance evaluation
    :param truth_df:
    :param pred_df:
    :return:
    """
    pred_df = pred_df.loc[truth_df.index]
    per_cell_measurement = defaultdict(dict)
    for cell in truth_df.index:
        drugs = sorted(truth_df.columns[~truth_df.loc[cell].isna()])
        pred_drugs = sorted(pred_df.columns[pred_df.loc[cell] != -1])
        effective_drugs = set(drugs) & set(pred_drugs)

        truth_vec = truth_df.loc[cell, effective_drugs]
        pred_vec = pred_df.loc[cell, effective_drugs]
        per_cell_measurement['auroc'][cell] = roc_auc_score(y_true=truth_vec, y_score=pred_vec)
        per_cell_measurement['postive_rate'][cell] = truth_vec.sum()/len(drugs)
        per_cell_measurement['aps'][cell] = average_precision_score(y_true=truth_vec, y_score=pred_vec)

    return pd.DataFrame.from_dict(per_cell_measurement)