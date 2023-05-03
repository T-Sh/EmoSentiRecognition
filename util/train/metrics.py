import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)


def multi_metrics(preds, y):
    _, y_pred_tags = torch.max(preds, dim=1)
    _, y_tags = torch.max(y, dim=1)

    y_pred_tags = y_pred_tags.cpu().data
    y_tags = y_tags.cpu().data

    acc = accuracy_score(y_tags, y_pred_tags, normalize=True)
    prec = precision_score(y_tags, y_pred_tags, average="weighted")
    f1 = f1_score(y_tags, y_pred_tags, average="weighted")
    rec = recall_score(y_tags, y_pred_tags, average="weighted")
    report = classification_report(y_tags, y_pred_tags, output_dict=True)

    return acc, prec, f1, rec, report


def multi_metrics_for_valid(y_pred_tags, y_tags):
    acc = accuracy_score(y_tags, y_pred_tags, normalize=True)
    prec = precision_score(y_tags, y_pred_tags, average="weighted")
    f1 = f1_score(y_tags, y_pred_tags, average="weighted")
    rec = recall_score(y_tags, y_pred_tags, average="weighted")
    report = classification_report(y_tags, y_pred_tags, output_dict=True)

    return acc, prec, f1, rec, report


def accuracy_7(out, labels):
    return np.sum(np.round(out) == np.round(labels)) / float(len(labels))


def multi_metrics_for_valid_with_confusion(y_pred_tags, y_tags):
    y_pred_tags = np.round(y_pred_tags)
    y_tags = np.round(y_tags)
    acc = accuracy_score(y_tags, y_pred_tags, normalize=True)
    prec = precision_score(y_tags, y_pred_tags, average="weighted")
    f1 = f1_score(y_tags, y_pred_tags, average="weighted")
    rec = recall_score(y_tags, y_pred_tags, average="weighted")
    report = classification_report(y_tags, y_pred_tags, output_dict=True)
    matrix = confusion_matrix(y_tags, y_pred_tags)

    acc7 = accuracy_7(y_pred_tags, y_tags)

    return acc, acc7, prec, f1, rec, report, matrix
