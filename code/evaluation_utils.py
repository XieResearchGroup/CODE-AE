import pandas as pd
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
    log_loss, auc, precision_recall_curve


def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)



def model_save_check(history, metric_name, tolerance_count=5, reset_count=1):
    save_flag = False
    stop_flag = False
    if 'best_index' not in history:
        history['best_index'] = 0
    if metric_name.endswith('loss'):
        if history[metric_name][-1] <= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1
    else:
        if history[metric_name][-1] >= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1

    if len(history[metric_name]) - history['best_index'] > tolerance_count * reset_count and history['best_index'] > 0:
        stop_flag = True

    return save_flag, stop_flag


def eval_ae_epoch(model, data_loader, device, history):
    model.eval()
    avg_loss_dict = defaultdict(float)
    for x_batch in data_loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history


def evaluate_target_classification_epoch(classifier, dataloader, device, history):
    y_truths = np.array([])
    y_preds = np.array([])
    classifier.eval()

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
            y_pred = torch.sigmoid(classifier(x_batch)).detach()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=y_preds))
    history['aps'].append(average_precision_score(y_true=y_truths, y_score=y_preds))
    history['f1'].append(f1_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['bce'].append(log_loss(y_true=y_truths, y_pred=y_preds))
    history['auprc'].append(auprc(y_true=y_truths, y_score=y_preds))

    return history


def evaluate_adv_classification_epoch(classifier, s_dataloader, t_dataloader, device, history):
    y_truths = np.array([])
    y_preds = np.array([])
    classifier.eval()

    for s_batch in s_dataloader:
        s_x = s_batch[0].to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, np.zeros(s_x.shape[0]).ravel()])
            s_y_pred = torch.sigmoid(classifier(s_x)).detach()
            y_preds = np.concatenate([y_preds, s_y_pred.cpu().detach().numpy().ravel()])

    for t_batch in t_dataloader:
        t_x = t_batch[0].to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, np.ones(t_x.shape[0]).ravel()])
            t_y_pred = torch.sigmoid(classifier(t_x)).detach()
            y_preds = np.concatenate([y_preds, t_y_pred.cpu().detach().numpy().ravel()])

    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=y_preds))
    history['aps'].append(average_precision_score(y_true=y_truths, y_score=y_preds))
    history['f1'].append(f1_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['bce'].append(log_loss(y_true=y_truths, y_pred=y_preds))
    history['auprc'].append(auprc(y_true=y_truths, y_score=y_preds))

    return history


def predict_target_classification(classifier, test_df, device):
    y_preds = np.array([])
    classifier.eval()

    for df in [test_df[i:i+64] for i in range(0,test_df.shape[0],64)]:
        x_batch = torch.from_numpy(df.values.astype('float32')).to(device)
        with torch.no_grad():
            y_pred = torch.sigmoid(classifier(x_batch)).detach()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    output_df = pd.DataFrame(y_preds,index=test_df.index,columns=['score'])

    return output_df